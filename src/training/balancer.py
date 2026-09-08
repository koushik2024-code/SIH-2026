"""
Class Balancing Module for Training Data Preparation.

Applies SMOTE (Synthetic Minority Over-sampling Technique) to address
class imbalance in the labeled fire dataset, with fallback to random
over-sampling for very small classes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class ClassBalancer:
    """
    SMOTE-based class balancer for fire classification training data.

    Falls back to RandomOverSampler when a class has fewer samples
    than the SMOTE k_neighbors threshold.
    """

    def __init__(self, sampling_strategy="auto", k_neighbors: int = 5, random_state: int = 42):
        """
        Parameters
        ----------
        sampling_strategy : str or dict
            Passed to SMOTE / RandomOverSampler.
            - "auto": resample all classes except majority to match majority.
            - dict: {class_label: desired_count}.
        k_neighbors : int
            Number of nearest neighbors for SMOTE (default 5).
        random_state : int
            Reproducibility seed.
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._balance_report: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def balance_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple:
        """
        Apply SMOTE oversampling to balance class distribution.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (numeric columns only).
        y : Series
            Target labels.

        Returns
        -------
        (X_balanced, y_balanced)
            Balanced feature matrix and label series.
        """
        if X.empty or len(y) == 0:
            logger.warning("Empty data -- nothing to balance.")
            return X, y

        # Encode labels for SMOTE (needs integer targets)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.values)

        original_dist = pd.Series(y).value_counts().to_dict()
        logger.info("Original class distribution:")
        for cls, count in sorted(original_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {cls:25s}  {count:>6d}")

        # Check minimum class size
        min_class_size = pd.Series(y_encoded).value_counts().min()

        X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0).values

        try:
            if min_class_size >= self.k_neighbors + 1:
                # Use SMOTE
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    sampling_strategy=self.sampling_strategy,
                    k_neighbors=self.k_neighbors,
                    random_state=self.random_state,
                )
                logger.info(f"Applying SMOTE (k_neighbors={self.k_neighbors}) ...")
                X_res, y_res = sampler.fit_resample(X_numeric, y_encoded)
                method_used = "SMOTE"
            else:
                # Fallback: adapt k_neighbors or use RandomOverSampler
                if min_class_size >= 2:
                    from imblearn.over_sampling import SMOTE
                    adapted_k = min(min_class_size - 1, self.k_neighbors)
                    sampler = SMOTE(
                        sampling_strategy=self.sampling_strategy,
                        k_neighbors=adapted_k,
                        random_state=self.random_state,
                    )
                    logger.info(
                        f"Applying SMOTE with adapted k_neighbors={adapted_k} "
                        f"(min class size = {min_class_size}) ..."
                    )
                    X_res, y_res = sampler.fit_resample(X_numeric, y_encoded)
                    method_used = f"SMOTE (k={adapted_k})"
                else:
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(
                        sampling_strategy=self.sampling_strategy,
                        random_state=self.random_state,
                    )
                    logger.info(
                        f"Falling back to RandomOverSampler "
                        f"(min class size = {min_class_size}) ..."
                    )
                    X_res, y_res = sampler.fit_resample(X_numeric, y_encoded)
                    method_used = "RandomOverSampler"

        except ImportError:
            logger.error(
                "imbalanced-learn is not installed. "
                "Install with: pip install imbalanced-learn"
            )
            return X, y
        except Exception as e:
            logger.error(f"Balancing failed: {e}. Returning original data.")
            return X, y

        # Decode labels back
        y_decoded = le.inverse_transform(y_res)
        y_balanced = pd.Series(y_decoded, name=y.name if hasattr(y, "name") else "fire_label")

        # Reconstruct DataFrame with original column names
        X_balanced = pd.DataFrame(X_res, columns=X.columns)

        balanced_dist = y_balanced.value_counts().to_dict()
        logger.info(f"Balanced class distribution (method: {method_used}):")
        for cls, count in sorted(balanced_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {cls:25s}  {count:>6d}")

        # Store report
        self._balance_report = {
            "method": method_used,
            "original_total": int(len(y)),
            "balanced_total": int(len(y_balanced)),
            "original_distribution": {str(k): int(v) for k, v in original_dist.items()},
            "balanced_distribution": {str(k): int(v) for k, v in balanced_dist.items()},
        }

        return X_balanced, y_balanced

    def get_balance_report(self) -> dict:
        """Return the most recent balance report."""
        return dict(self._balance_report)
