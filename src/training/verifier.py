"""
Manual Verification Module for Training Data Quality Assurance.

Generates stratified samples of auto-labeled data for human review,
exports them as annotatable CSVs, and computes inter-rater agreement
metrics once the reviewer has filled in corrections.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ManualVerifier:
    """
    Stratified sampling and agreement-metric calculator for
    manual verification of semi-auto-labeled fire data.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._agreement_metrics: dict = {}

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def generate_review_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 50,
        stratify_col: str = "fire_label",
    ) -> pd.DataFrame:
        """
        Draw a stratified random sample for manual review.

        Ensures every class gets at least 1 sample. Classes with
        fewer records than their proportional share contribute all
        their records.
        """
        if df.empty or stratify_col not in df.columns:
            logger.warning("Cannot generate sample -- empty data or missing column.")
            return pd.DataFrame()

        n = len(df)
        sample_size = min(sample_size, n)

        class_counts = df[stratify_col].value_counts()
        n_classes = len(class_counts)

        # Proportional allocation with at least 1 per class
        allocations: dict = {}
        remaining = sample_size

        for cls, count in class_counts.items():
            share = max(1, int(round(sample_size * count / n)))
            share = min(share, count, remaining)
            allocations[cls] = share
            remaining -= share
            if remaining <= 0:
                break

        # Distribute any leftover budget to largest classes
        if remaining > 0:
            for cls in class_counts.index:
                available = class_counts[cls] - allocations.get(cls, 0)
                add = min(remaining, available)
                allocations[cls] = allocations.get(cls, 0) + add
                remaining -= add
                if remaining <= 0:
                    break

        samples = []
        rng = np.random.RandomState(self.random_state)
        for cls, alloc in allocations.items():
            cls_df = df[df[stratify_col] == cls]
            if alloc >= len(cls_df):
                samples.append(cls_df)
            else:
                samples.append(cls_df.sample(n=alloc, random_state=rng))

        sample_df = pd.concat(samples, ignore_index=False).sort_index()

        logger.info(
            f"Generated review sample: {len(sample_df)} records "
            f"across {n_classes} classes"
        )
        for cls, alloc in allocations.items():
            logger.info(f"  {cls:25s}  {alloc:>4d} samples")

        return sample_df

    # ------------------------------------------------------------------
    # CSV export / import
    # ------------------------------------------------------------------

    def export_review_csv(self, sample_df: pd.DataFrame, output_path) -> Path:
        """Export the review sample as a CSV with reviewer annotation columns."""
        output_path = Path(output_path)

        export_cols = []
        for col in [
            "fire_id", "latitude", "longitude",
            "fire_label", "label_confidence",
            "brightness", "frp", "land_cover",
            "nearest_facility_type", "distance_to_nearest_industrial",
            "cluster_class", "duration_days",
        ]:
            if col in sample_df.columns:
                export_cols.append(col)

        out = sample_df[export_cols].copy()
        out["reviewer_label"] = ""
        out["reviewer_notes"] = ""

        out.to_csv(output_path, index=False)
        logger.info(f"Exported review sample ({len(out)} rows) -> {output_path}")
        return output_path

    def import_reviewed_csv(self, path) -> pd.DataFrame:
        """Read back a reviewer-annotated CSV and compute agreement metrics."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Reviewed CSV not found: {path}")

        reviewed = pd.read_csv(path)
        logger.info(f"Imported reviewed CSV: {len(reviewed)} rows")

        has_review = (
            reviewed["reviewer_label"].notna()
            & (reviewed["reviewer_label"].astype(str).str.strip() != "")
        )
        n_reviewed = has_review.sum()

        if n_reviewed > 0:
            original = reviewed.loc[has_review, "fire_label"].astype(str).values
            reviewed_labels = reviewed.loc[has_review, "reviewer_label"].astype(str).values
            self._agreement_metrics = self.compute_agreement_metrics(
                original, reviewed_labels
            )
            logger.info(
                f"Agreement computed on {n_reviewed} reviewed samples: "
                f"accuracy={self._agreement_metrics['accuracy']:.2%}, "
                f"kappa={self._agreement_metrics['cohens_kappa']:.3f}"
            )
        else:
            logger.warning("No reviewer labels found in imported CSV.")

        return reviewed

    # ------------------------------------------------------------------
    # Agreement metrics
    # ------------------------------------------------------------------

    def compute_agreement_metrics(self, original_labels, reviewed_labels) -> dict:
        """
        Compute inter-rater agreement between auto and human labels.

        Returns accuracy, Cohen's Kappa, and per-class precision/recall.
        """
        original_labels = np.asarray(original_labels)
        reviewed_labels = np.asarray(reviewed_labels)
        n = len(original_labels)

        if n == 0:
            return {"accuracy": 0.0, "cohens_kappa": 0.0, "per_class": {}}

        # Overall accuracy
        accuracy = np.mean(original_labels == reviewed_labels)

        # Cohen's Kappa
        labels = np.unique(np.concatenate([original_labels, reviewed_labels]))
        n_labels = len(labels)

        if n_labels <= 1:
            kappa = 1.0 if accuracy == 1.0 else 0.0
        else:
            po = accuracy
            pe = 0.0
            for lbl in labels:
                p_orig = np.mean(original_labels == lbl)
                p_rev = np.mean(reviewed_labels == lbl)
                pe += p_orig * p_rev
            kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0

        # Per-class precision / recall (reviewer as ground truth)
        per_class: dict = {}
        for lbl in labels:
            tp = np.sum((original_labels == lbl) & (reviewed_labels == lbl))
            fp = np.sum((original_labels == lbl) & (reviewed_labels != lbl))
            fn = np.sum((original_labels != lbl) & (reviewed_labels == lbl))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[str(lbl)] = {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
            }

        metrics = {
            "accuracy": round(float(accuracy), 4),
            "cohens_kappa": round(float(kappa), 4),
            "n_reviewed": int(n),
            "per_class": per_class,
        }
        self._agreement_metrics = metrics
        return metrics

    def get_agreement_metrics(self) -> dict:
        """Return most recent agreement metrics."""
        return dict(self._agreement_metrics)
