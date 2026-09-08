"""
Training Data Preparation Pipeline.

Orchestrates the full training data preparation workflow:
    1. Load enriched fire data (from Part 2.4 output)
    2. Semi-Automatic Labeling
    3. Manual Review Sample Generation
    4. Feature Selection
    5. SMOTE Class Balancing
    6. Export final training dataset + metadata
"""

import json
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

from config import settings
from .labeler import SemiAutoLabeler
from .verifier import ManualVerifier
from .feature_selector import FeatureSelector
from .balancer import ClassBalancer

logger = logging.getLogger(__name__)


class TrainingDataPipeline:
    """
    End-to-end pipeline for preparing a labeled, balanced, feature-selected
    training dataset for supervised fire classification.
    """

    def __init__(
        self,
        proximity_threshold_km: float = 1.0,
        review_sample_size: int = 50,
        correlation_threshold: float = 0.95,
        top_k_features: int | None = None,
        smote_strategy: str = "auto",
    ):
        self.labeler = SemiAutoLabeler(proximity_threshold_km=proximity_threshold_km)
        self.verifier = ManualVerifier()
        self.selector = FeatureSelector(
            correlation_threshold=correlation_threshold,
            top_k=top_k_features,
        )
        self.balancer = ClassBalancer(sampling_strategy=smote_strategy)
        self.review_sample_size = review_sample_size

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the full training data preparation pipeline.

        Returns
        -------
        DataFrame
            The final balanced, feature-selected training dataset.
        """
        logger.info("=" * 65)
        logger.info("  TRAINING DATA PREPARATION PIPELINE (Part 3)")
        logger.info("=" * 65)

        output_dir = settings.PROCESSED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # ==============================================================
        # Step 0: Load enriched data
        # ==============================================================
        logger.info("[0/5] Loading enriched fire data ...")

        # Try Part 2.4 master output first, then fall back to Part 1 output
        master_path = output_dir / "master_enriched_fires.csv"
        enriched_path = output_dir / "enriched_fire_data.csv"

        if master_path.exists():
            fires_df = pd.read_csv(master_path)
            logger.info(f"Loaded master enriched data: {len(fires_df)} records from {master_path}")
        elif enriched_path.exists():
            fires_df = pd.read_csv(enriched_path)
            logger.info(f"Loaded enriched data: {len(fires_df)} records from {enriched_path}")
        else:
            logger.error(
                "No enriched fire data found. Run Part 1 (and optionally Part 2.4) first.\n"
                f"  Expected: {master_path}\n"
                f"       or: {enriched_path}"
            )
            return pd.DataFrame()

        if fires_df.empty:
            logger.error("Fire dataset is empty. Cannot prepare training data.")
            return pd.DataFrame()

        # Load facilities if available
        facilities_path = output_dir / "industrial_facilities.geojson"
        facilities_gdf = None
        if facilities_path.exists():
            facilities_gdf = gpd.read_file(facilities_path)
            logger.info(f"Loaded {len(facilities_gdf)} industrial facilities")

        # ==============================================================
        # Step 1: Semi-Automatic Labeling
        # ==============================================================
        logger.info("[1/5] Running Semi-Automatic Labeling ...")
        labeled_df = self.labeler.label_fires(fires_df, facilities_gdf)

        # Save labeled (pre-balance) dataset
        labeled_path = output_dir / "labeled_fire_data.csv"
        labeled_df.to_csv(labeled_path, index=False)
        logger.info(f"Saved labeled data -> {labeled_path}")

        # ==============================================================
        # Step 2: Manual Review Sample
        # ==============================================================
        logger.info("[2/5] Generating Manual Review Sample ...")
        sample_df = self.verifier.generate_review_sample(
            labeled_df,
            sample_size=self.review_sample_size,
            stratify_col="fire_label",
        )

        review_path = output_dir / "review_sample.csv"
        if not sample_df.empty:
            self.verifier.export_review_csv(sample_df, review_path)

        # ==============================================================
        # Step 3: Feature Selection
        # ==============================================================
        logger.info("[3/5] Running Statistical Feature Selection ...")
        selected_df, feature_report = self.selector.select_features(
            labeled_df, target_col="fire_label"
        )

        # Save feature importance chart
        chart_path = output_dir / "feature_importance.png"
        self.selector.plot_feature_importance(str(chart_path))

        # ==============================================================
        # Step 4: SMOTE Class Balancing
        # ==============================================================
        logger.info("[4/5] Applying SMOTE Class Balancing ...")

        if "fire_label" in selected_df.columns:
            feature_cols = [c for c in selected_df.columns if c != "fire_label"]
            X = selected_df[feature_cols]
            y = selected_df["fire_label"]

            X_balanced, y_balanced = self.balancer.balance_dataset(X, y)

            # Reconstruct final dataset
            final_df = X_balanced.copy()
            final_df["fire_label"] = y_balanced.values
        else:
            logger.warning("No fire_label column after feature selection.")
            final_df = selected_df

        # ==============================================================
        # Step 5: Save Outputs
        # ==============================================================
        logger.info("[5/5] Saving final training dataset ...")

        # Training dataset
        training_path = output_dir / "training_dataset.csv"
        final_df.to_csv(training_path, index=False)
        logger.info(f"Saved training dataset ({len(final_df)} rows) -> {training_path}")

        # Metadata JSON
        metadata = {
            "pipeline": "Training Data Preparation (Part 3)",
            "input_records": len(fires_df),
            "labeled_records": len(labeled_df),
            "final_training_records": len(final_df),
            "label_distribution_original": self.labeler.get_label_distribution(),
            "label_distribution_balanced": self.balancer.get_balance_report().get(
                "balanced_distribution", {}
            ),
            "feature_selection": {
                "n_candidates": feature_report.get("n_candidates", 0),
                "n_selected": feature_report.get("n_selected", 0),
                "selected_features": feature_report.get("selected_features", []),
                "dropped_correlated": feature_report.get("dropped_correlated", []),
            },
            "balance_report": self.balancer.get_balance_report(),
            "review_sample_size": len(sample_df) if not sample_df.empty else 0,
        }

        metadata_path = output_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        logger.info(f"Saved training metadata -> {metadata_path}")

        # ==============================================================
        # Summary
        # ==============================================================
        logger.info("=" * 65)
        logger.info("  TRAINING DATA PREPARATION COMPLETE")
        logger.info("=" * 65)
        logger.info(f"  Input records:     {len(fires_df):>8d}")
        logger.info(f"  Labeled records:   {len(labeled_df):>8d}")
        logger.info(f"  Training records:  {len(final_df):>8d}")
        logger.info(f"  Features selected: {feature_report.get('n_selected', '?'):>8}")
        logger.info(f"  Output files:")
        logger.info(f"    {training_path}")
        logger.info(f"    {metadata_path}")
        logger.info(f"    {review_path}")
        logger.info(f"    {chart_path}")
        logger.info("=" * 65)

        return final_df
