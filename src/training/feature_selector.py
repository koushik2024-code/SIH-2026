"""
Feature Selection Module for Training Data Preparation.

Performs statistical analysis to identify the most predictive features
for fire classification using ANOVA F-test, Mutual Information, and
correlation-based filtering.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default candidate features that may exist in the enriched dataset
CANDIDATE_FEATURES = [
    "brightness", "frp", "confidence", "scan", "track",
    "brightness_frp_ratio",
    "hour_of_day", "day_of_week", "month", "is_daytime",
    "distance_to_nearest_industrial", "is_near_industrial",
    "fire_cluster_id", "duration_days",
    "gi_zscore", "gi_pvalue", "is_hotspot",
    "land_cover", "label_confidence",
]


class FeatureSelector:
    """
    Statistical feature selection for fire classification training data.

    Combines ANOVA F-test scores, Mutual Information scores, and
    pairwise correlation filtering to rank and select the most
    informative feature subset.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        top_k: int | None = None,
    ):
        """
        Parameters
        ----------
        correlation_threshold : float
            Drop one of any pair of features with Pearson |r| above this.
        top_k : int, optional
            If set, keep only the top-k features by combined rank.
            If None, keep all features that pass correlation filtering.
        """
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self._report: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = "fire_label",
        candidate_features: list[str] | None = None,
    ) -> tuple:
        """
        Run the full feature-selection pipeline.

        Parameters
        ----------
        df : DataFrame
            Labeled dataset with numerical features and a target column.
        target_col : str
            Name of the label column.
        candidate_features : list[str], optional
            Override the default candidate feature list.

        Returns
        -------
        (selected_df, report)
            selected_df : DataFrame with target + selected features only.
            report : dict with feature rankings and scores.
        """
        if df.empty or target_col not in df.columns:
            logger.warning("Cannot select features -- empty data or missing target.")
            return df, {}

        candidates = candidate_features or CANDIDATE_FEATURES

        # Filter to features actually present and numeric
        available = [c for c in candidates if c in df.columns]
        numeric_df = df[available].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")

        feature_names = list(numeric_df.columns)
        if len(feature_names) == 0:
            logger.warning("No numeric features available for selection.")
            return df, {}

        X = numeric_df.fillna(0).values
        y_raw = df[target_col].values

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        logger.info(f"Running feature selection on {len(feature_names)} candidates ...")

        # ----------------------------------------------------------
        # 1. ANOVA F-test
        # ----------------------------------------------------------
        f_scores, f_pvalues = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)

        # ----------------------------------------------------------
        # 2. Mutual Information
        # ----------------------------------------------------------
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # ----------------------------------------------------------
        # 3. Combined ranking (lower rank = better)
        # ----------------------------------------------------------
        f_ranks = np.argsort(np.argsort(-f_scores))  # descending
        mi_ranks = np.argsort(np.argsort(-mi_scores))
        combined_rank = 0.5 * f_ranks + 0.5 * mi_ranks

        ranking = sorted(
            zip(feature_names, f_scores, mi_scores, combined_rank, f_pvalues),
            key=lambda x: x[3],
        )

        logger.info("Feature ranking (best first):")
        logger.info(f"  {'Feature':30s}  {'F-score':>10s}  {'MI':>10s}  {'Rank':>6s}  {'p-val':>10s}")
        for feat, fs, mi, cr, pv in ranking:
            logger.info(f"  {feat:30s}  {fs:10.2f}  {mi:10.4f}  {cr:6.1f}  {pv:10.4e}")

        # ----------------------------------------------------------
        # 4. Correlation filtering
        # ----------------------------------------------------------
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        drop_cols = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > self.correlation_threshold].tolist()
            if correlated:
                # Drop the one with worse combined rank
                for corr_col in correlated:
                    idx_col = feature_names.index(col) if col in feature_names else 999
                    idx_corr = feature_names.index(corr_col) if corr_col in feature_names else 999
                    rank_col = combined_rank[idx_col] if idx_col < len(combined_rank) else 999
                    rank_corr = combined_rank[idx_corr] if idx_corr < len(combined_rank) else 999
                    if rank_col > rank_corr:
                        drop_cols.add(col)
                    else:
                        drop_cols.add(corr_col)

        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} correlated features: {drop_cols}")

        selected_features = [f for f, _, _, _, _ in ranking if f not in drop_cols]

        # Apply top_k if specified
        if self.top_k is not None and len(selected_features) > self.top_k:
            selected_features = selected_features[: self.top_k]
            logger.info(f"Keeping top {self.top_k} features")

        # ----------------------------------------------------------
        # Build report
        # ----------------------------------------------------------
        self._report = {
            "n_candidates": len(feature_names),
            "n_selected": len(selected_features),
            "n_dropped_correlation": len(drop_cols),
            "selected_features": selected_features,
            "dropped_correlated": list(drop_cols),
            "feature_scores": [
                {
                    "feature": feat,
                    "f_score": round(float(fs), 4),
                    "mutual_info": round(float(mi), 4),
                    "combined_rank": round(float(cr), 1),
                    "f_pvalue": float(pv),
                }
                for feat, fs, mi, cr, pv in ranking
            ],
        }

        # Build selected DataFrame
        keep_cols = selected_features + [target_col]
        available_keep = [c for c in keep_cols if c in df.columns]
        selected_df = df[available_keep].copy()

        logger.info(
            f"Feature selection complete: {len(feature_names)} -> "
            f"{len(selected_features)} features"
        )

        return selected_df, self._report

    def get_feature_importance_report(self) -> dict:
        """Return the most recent feature importance report."""
        return dict(self._report)

    def plot_feature_importance(self, output_path=None):
        """Save a bar chart of feature importance scores."""
        if not self._report or "feature_scores" not in self._report:
            logger.warning("No feature report available. Run select_features first.")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            scores = self._report["feature_scores"]
            features = [s["feature"] for s in scores]
            f_vals = [s["f_score"] for s in scores]
            mi_vals = [s["mutual_info"] for s in scores]

            # Normalize for side-by-side comparison
            f_max = max(f_vals) if max(f_vals) > 0 else 1
            mi_max = max(mi_vals) if max(mi_vals) > 0 else 1
            f_norm = [v / f_max for v in f_vals]
            mi_norm = [v / mi_max for v in mi_vals]

            fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.4)))
            y_pos = np.arange(len(features))
            bar_height = 0.35

            ax.barh(y_pos - bar_height / 2, f_norm, bar_height,
                    label="ANOVA F-score (norm)", color="#2196F3", alpha=0.8)
            ax.barh(y_pos + bar_height / 2, mi_norm, bar_height,
                    label="Mutual Information (norm)", color="#FF9800", alpha=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel("Normalized Score")
            ax.set_title("Feature Importance: ANOVA F-test vs Mutual Information")
            ax.legend(loc="lower right")
            ax.invert_yaxis()
            plt.tight_layout()

            if output_path is None:
                output_path = "feature_importance.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Feature importance chart saved -> {output_path}")

        except ImportError:
            logger.warning("matplotlib not available. Skipping plot generation.")
