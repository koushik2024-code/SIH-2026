"""
Model Evaluation Module for Fire Classification.

Provides comprehensive evaluation metrics, confusion matrix heatmaps,
ROC-AUC curves, and model comparison utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics, plots, and comparisons.
    """

    def __init__(self, class_names: list = None):
        """
        Parameters
        ----------
        class_names : list, optional
            Human-readable class names for display in reports and plots.
        """
        self.class_names = class_names or []
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, model, X_test, y_test, model_name: str = "model") -> dict:
        """
        Run the full evaluation suite on a trained model.

        Parameters
        ----------
        model : estimator
            Trained sklearn-compatible classifier.
        X_test : array-like
            Test feature matrix.
        y_test : array-like
            True test labels (integer-encoded).
        model_name : str
            Name for this model (used in filenames and reports).

        Returns
        -------
        dict
            Complete evaluation metrics.
        """
        logger.info(f"Evaluating model: {model_name} ...")

        y_pred = model.predict(X_test)

        # Get probabilities if available
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        per_class = self._per_class_metrics(y_test, y_pred)
        report_str = self._classification_report(y_test, y_pred)

        result = {
            "model_name": model_name,
            "overall_metrics": metrics,
            "per_class_metrics": per_class,
            "classification_report": report_str,
        }

        self._results[model_name] = result

        # Log summary
        logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"  Weighted F1:       {metrics['f1_weighted']:.4f}")
        logger.info(f"  Macro F1:          {metrics['f1_macro']:.4f}")
        if metrics.get("roc_auc") is not None:
            logger.info(f"  ROC-AUC (weighted): {metrics['roc_auc']:.4f}")
        logger.info(f"\n{report_str}")

        return result

    def plot_confusion_matrix(self, model, X_test, y_test, output_path, model_name="model"):
        """Generate and save a confusion matrix heatmap."""
        if not HAS_PLOT:
            logger.warning("matplotlib/seaborn not available. Skipping confusion matrix plot.")
            return

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        labels = self.class_names if self.class_names else [str(i) for i in range(cm.shape[0])]
        # Truncate labels if they don't match
        if len(labels) != cm.shape[0]:
            labels = [str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved -> {output_path}")

    def plot_roc_curves(self, model, X_test, y_test, output_path, model_name="model"):
        """Generate and save one-vs-rest ROC curves for each class."""
        if not HAS_PLOT:
            logger.warning("matplotlib/seaborn not available. Skipping ROC plot.")
            return

        if not hasattr(model, "predict_proba"):
            logger.warning(f"Model {model_name} does not support predict_proba. Skipping ROC.")
            return

        y_prob = model.predict_proba(X_test)
        n_classes = y_prob.shape[1]
        classes = np.arange(n_classes)

        # Binarize labels for one-vs-rest
        y_bin = label_binarize(y_test, classes=classes)
        if y_bin.shape[1] == 1:
            # Binary case
            y_bin = np.hstack([1 - y_bin, y_bin])

        labels = self.class_names if len(self.class_names) == n_classes else [f"Class {i}" for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            if y_bin[:, i].sum() == 0:
                continue
            try:
                auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
                # Compute ROC manually
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f"{labels[i]} (AUC = {auc:.3f})")
            except Exception as e:
                logger.warning(f"ROC for class {i} failed: {e}")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves (One-vs-Rest) - {model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curves saved -> {output_path}")

    def plot_model_comparison(self, results: dict, output_path):
        """Generate a side-by-side bar chart comparing model metrics."""
        if not HAS_PLOT:
            logger.warning("matplotlib not available. Skipping comparison plot.")
            return

        model_names = list(results.keys())
        metric_keys = ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]
        metric_labels = ["Accuracy", "F1 (Weighted)", "F1 (Macro)", "Precision", "Recall"]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metric_labels))
        width = 0.8 / len(model_names)
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

        for i, name in enumerate(model_names):
            metrics = results[name].get("overall_metrics", {})
            values = [metrics.get(k, 0) for k in metric_keys]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=name.upper(),
                         color=colors[i % len(colors)], alpha=0.85)
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Model comparison chart saved -> {output_path}")

    def compare_models(self, results: dict) -> str:
        """
        Compare models and return the name of the best one.

        Uses weighted F1-score as the primary selection criterion.
        """
        best_name = None
        best_f1 = -1

        logger.info("\n" + "=" * 60)
        logger.info("  MODEL COMPARISON")
        logger.info("=" * 60)
        logger.info(f"  {'Model':<12s}  {'Accuracy':>10s}  {'F1-Weighted':>12s}  {'F1-Macro':>10s}")
        logger.info("-" * 60)

        for name, res in results.items():
            m = res.get("overall_metrics", {})
            acc = m.get("accuracy", 0)
            f1w = m.get("f1_weighted", 0)
            f1m = m.get("f1_macro", 0)
            logger.info(f"  {name.upper():<12s}  {acc:>10.4f}  {f1w:>12.4f}  {f1m:>10.4f}")

            if f1w > best_f1:
                best_f1 = f1w
                best_name = name

        logger.info("-" * 60)
        logger.info(f"  Best model: {best_name.upper()} (F1-Weighted = {best_f1:.4f})")
        logger.info("=" * 60)

        return best_name

    def export_results(self, results: dict, output_path):
        """Save all evaluation results to a JSON file."""
        output_path = Path(output_path)

        # Make JSON-serializable
        serializable = {}
        for name, res in results.items():
            serializable[name] = {
                "model_name": res.get("model_name", name),
                "overall_metrics": res.get("overall_metrics", {}),
                "per_class_metrics": res.get("per_class_metrics", {}),
                "classification_report": res.get("classification_report", ""),
            }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=4, default=str)
        logger.info(f"Evaluation results saved -> {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, y_true, y_pred, y_prob=None) -> dict:
        """Compute overall classification metrics."""
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        }

        # ROC-AUC (requires probabilities)
        if y_prob is not None:
            try:
                n_classes = y_prob.shape[1]
                if n_classes == 2:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    auc = roc_auc_score(
                        y_true, y_prob,
                        multi_class="ovr", average="weighted"
                    )
                metrics["roc_auc"] = round(float(auc), 4)
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        return metrics

    def _per_class_metrics(self, y_true, y_pred) -> dict:
        """Compute per-class precision, recall, F1, support."""
        labels = np.unique(np.concatenate([y_true, y_pred]))
        names = self.class_names if len(self.class_names) == len(labels) else [str(l) for l in labels]

        per_class = {}
        for i, label in enumerate(labels):
            mask_true = y_true == label
            mask_pred = y_pred == label
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)
            support = int(np.sum(mask_true))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            class_name = names[i] if i < len(names) else str(label)
            per_class[class_name] = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "support": support,
            }

        return per_class

    def _classification_report(self, y_true, y_pred) -> str:
        """Generate sklearn classification report string."""
        labels = np.unique(np.concatenate([y_true, y_pred]))
        names = self.class_names if len(self.class_names) == len(labels) else None
        return classification_report(y_true, y_pred, target_names=names, zero_division=0)
