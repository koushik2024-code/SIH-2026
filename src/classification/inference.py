"""
Inference Pipeline for Fire Classification.

Loads a trained model and feature configuration, accepts new fire data,
runs it through the feature pipeline, and returns classified results
with confidence scores.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production inference pipeline for classifying new fire data.
    """

    def __init__(self, model_path: str = None, metadata_path: str = None):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to the persisted model (.pkl).
        metadata_path : str, optional
            Path to model_metadata.json containing feature list and class mapping.
        """
        self.model = None
        self.feature_names: list = []
        self.class_names: list = []
        self.class_id_map: dict = {}
        self._metadata: dict = {}

        if model_path:
            self.load(model_path, metadata_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: str, metadata_path: str = None):
        """
        Load a trained model and its metadata.

        Parameters
        ----------
        model_path : str
            Path to joblib-persisted model.
        metadata_path : str, optional
            Path to model_metadata.json.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        if metadata_path:
            meta_path = Path(metadata_path)
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    self._metadata = json.load(f)
                self.feature_names = self._metadata.get("feature_names", [])
                self.class_names = self._metadata.get("class_names", [])
                self.class_id_map = self._metadata.get("class_id_map", {})
                logger.info(f"Metadata loaded: {len(self.feature_names)} features, "
                           f"{len(self.class_names)} classes")

    def predict(self, fire_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify fire records and return results with predictions.

        Parameters
        ----------
        fire_df : DataFrame
            Fire data with feature columns.

        Returns
        -------
        DataFrame
            Original data augmented with ``predicted_class``,
            ``predicted_label``, and ``prediction_confidence`` columns.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        if fire_df.empty:
            logger.warning("Empty DataFrame -- nothing to predict.")
            return fire_df

        logger.info(f"Running inference on {len(fire_df)} records ...")

        # Prepare features
        X = self._prepare_features(fire_df)

        # Predict
        predictions = self.model.predict(X)

        # Get confidence scores
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = np.ones(len(predictions)) * 0.5

        # Build result DataFrame
        result_df = fire_df.copy()
        result_df["predicted_class"] = predictions

        # Map class IDs to labels
        if self.class_names:
            result_df["predicted_label"] = [
                self.class_names[int(p)] if int(p) < len(self.class_names) else f"Unknown ({p})"
                for p in predictions
            ]
        else:
            result_df["predicted_label"] = predictions

        result_df["prediction_confidence"] = np.round(confidence, 4)

        # Summary
        label_col = "predicted_label"
        dist = result_df[label_col].value_counts()
        logger.info("Prediction distribution:")
        for label, count in dist.items():
            pct = count / len(result_df) * 100
            logger.info(f"  {str(label):25s}  {count:>6d}  ({pct:5.1f}%)")

        return result_df

    def predict_single(self, feature_dict: dict) -> dict:
        """
        Classify a single fire record.

        Parameters
        ----------
        feature_dict : dict
            Feature name -> value mapping.

        Returns
        -------
        dict
            {"predicted_class", "predicted_label", "confidence", "probabilities"}
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        df = pd.DataFrame([feature_dict])
        X = self._prepare_features(df)

        prediction = int(self.model.predict(X)[0])

        result = {
            "predicted_class": prediction,
            "predicted_label": (
                self.class_names[prediction]
                if prediction < len(self.class_names)
                else f"Unknown ({prediction})"
            ),
        }

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            result["confidence"] = round(float(np.max(probs)), 4)
            result["probabilities"] = {
                self.class_names[i] if i < len(self.class_names) else f"Class {i}": round(float(p), 4)
                for i, p in enumerate(probs)
            }
        else:
            result["confidence"] = 0.5
            result["probabilities"] = {}

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select, order, and clean features to match training schema."""
        if self.feature_names:
            # Use only known features, in the correct order
            missing = [f for f in self.feature_names if f not in df.columns]
            if missing:
                logger.warning(f"Missing features (filled with 0): {missing}")

            X = pd.DataFrame()
            for feat in self.feature_names:
                if feat in df.columns:
                    X[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)
                else:
                    X[feat] = 0.0
        else:
            # Fall back to all numeric columns
            X = df.select_dtypes(include=[np.number]).fillna(0)

        return X.values
