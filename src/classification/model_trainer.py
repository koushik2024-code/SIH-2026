"""
Model Training Module for Fire Classification.

Trains Random Forest and XGBoost classifiers with hyperparameter
tuning via RandomizedSearchCV / GridSearchCV, 5-fold stratified
cross-validation, and model persistence with joblib.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Only Random Forest will be available.")


class ModelTrainer:
    """
    Dual-model trainer for fire classification using Random Forest
    and XGBoost with hyperparameter tuning.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        n_iter: int = 50,
        search_method: str = "random",
        random_state: int = 42,
        scoring: str = "f1_weighted",
    ):
        """
        Parameters
        ----------
        test_size : float
            Fraction of data reserved for testing (default 0.2).
        cv_folds : int
            Number of cross-validation folds (default 5).
        n_iter : int
            Number of parameter settings sampled for RandomizedSearchCV.
        search_method : str
            "random" for RandomizedSearchCV, "grid" for GridSearchCV.
        random_state : int
            Reproducibility seed.
        scoring : str
            Scoring metric for CV (default "f1_weighted").
        """
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.search_method = search_method
        self.random_state = random_state
        self.scoring = scoring

        self._label_encoder = LabelEncoder()
        self._feature_names: list = []
        self._class_names: list = []
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train both Random Forest and XGBoost models.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (numeric columns).
        y : Series
            Target labels (string class names).

        Returns
        -------
        dict
            Keys: "rf", "xgb" (if available), each containing
            {"model", "best_params", "cv_score", "X_test", "y_test"}.
        """
        logger.info("=" * 65)
        logger.info("  MODEL TRAINING PIPELINE")
        logger.info("=" * 65)

        self._feature_names = list(X.columns)

        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y.values)
        self._class_names = list(self._label_encoder.classes_)

        logger.info(f"Features: {len(self._feature_names)}")
        logger.info(f"Classes:  {self._class_names}")
        logger.info(f"Samples:  {len(X)}")

        # Fill NaN and ensure numeric
        X_clean = X.apply(pd.to_numeric, errors="coerce").fillna(0).values

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_encoded,
            test_size=self.test_size,
            stratify=y_encoded,
            random_state=self.random_state,
        )

        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

        results = {}

        # Train Random Forest
        logger.info("-" * 50)
        logger.info("Training Random Forest Classifier ...")
        rf_model, rf_params, rf_cv = self._train_random_forest(X_train, y_train)
        results["rf"] = {
            "model": rf_model,
            "best_params": rf_params,
            "cv_score": rf_cv,
            "X_test": X_test,
            "y_test": y_test,
        }

        # Train XGBoost
        if HAS_XGBOOST:
            logger.info("-" * 50)
            logger.info("Training XGBoost Classifier ...")
            xgb_model, xgb_params, xgb_cv = self._train_xgboost(X_train, y_train)
            results["xgb"] = {
                "model": xgb_model,
                "best_params": xgb_params,
                "cv_score": xgb_cv,
                "X_test": X_test,
                "y_test": y_test,
            }
        else:
            logger.warning("XGBoost not available. Skipping.")

        self._results = results
        return results

    def get_feature_names(self) -> list:
        """Return the feature names used during training."""
        return list(self._feature_names)

    def get_class_names(self) -> list:
        """Return the class names from the label encoder."""
        return list(self._class_names)

    def get_label_encoder(self) -> LabelEncoder:
        """Return the fitted label encoder."""
        return self._label_encoder

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(model, path):
        """Save a trained model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved -> {path}")

    @staticmethod
    def load_model(path):
        """Load a trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        logger.info(f"Model loaded <- {path}")
        return model

    # ------------------------------------------------------------------
    # Private: Random Forest
    # ------------------------------------------------------------------

    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest with hyperparameter tuning."""
        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample"],
            "max_features": ["sqrt", "log2"],
        }

        base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        if self.search_method == "grid":
            search = GridSearchCV(
                base_model, param_distributions,
                cv=cv, scoring=self.scoring,
                n_jobs=-1, verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_distributions,
                n_iter=min(self.n_iter, self._count_combinations(param_distributions)),
                cv=cv, scoring=self.scoring,
                n_jobs=-1, verbose=1,
                random_state=self.random_state,
            )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

        logger.info(f"RF Best CV {self.scoring}: {best_cv_score:.4f}")
        logger.info(f"RF Best params: {best_params}")

        # Feature importance
        importances = best_model.feature_importances_
        feat_imp = sorted(
            zip(self._feature_names, importances),
            key=lambda x: -x[1]
        )
        logger.info("RF Feature importance (top 10):")
        for feat, imp in feat_imp[:10]:
            logger.info(f"  {feat:30s}  {imp:.4f}")

        return best_model, best_params, best_cv_score

    # ------------------------------------------------------------------
    # Private: XGBoost
    # ------------------------------------------------------------------

    def _train_xgboost(self, X_train, y_train):
        """Train XGBoost with hyperparameter tuning."""
        n_classes = len(np.unique(y_train))

        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
        }

        base_model = XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        if self.search_method == "grid":
            search = GridSearchCV(
                base_model, param_distributions,
                cv=cv, scoring=self.scoring,
                n_jobs=-1, verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_distributions,
                n_iter=min(self.n_iter, self._count_combinations(param_distributions)),
                cv=cv, scoring=self.scoring,
                n_jobs=-1, verbose=1,
                random_state=self.random_state,
            )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

        logger.info(f"XGB Best CV {self.scoring}: {best_cv_score:.4f}")
        logger.info(f"XGB Best params: {best_params}")

        # Feature importance
        importances = best_model.feature_importances_
        feat_imp = sorted(
            zip(self._feature_names, importances),
            key=lambda x: -x[1]
        )
        logger.info("XGB Feature importance (top 10):")
        for feat, imp in feat_imp[:10]:
            logger.info(f"  {feat:30s}  {imp:.4f}")

        return best_model, best_params, best_cv_score

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _count_combinations(param_dict):
        """Count total combinations in a parameter grid."""
        total = 1
        for values in param_dict.values():
            total *= len(values)
        return total
