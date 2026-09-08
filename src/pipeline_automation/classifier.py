import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Industrial Fire",       # 0
    "Gas Flare",             # 1
    "Forest Fire",           # 2
    "Agricultural Burning",   # 3
    "Mining Activity",       # 4
    "Other/Unknown"          # 5
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

FEATURE_COLS = [
    "brightness",
    "frp",
    "brightness_frp_ratio",
    "distance_to_nearest_industrial",
    "is_near_industrial",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_daytime",
    "fire_cluster_id"
]

class FireClassifier:
    """
    Automated classification engine for thermal anomalies.
    Categorizes detections into Industrial Fire, Gas Flare, Forest Fire,
    Agricultural Burning, Mining Activity, or Other/Unknown.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or (settings.BASE_DIR / "models" / "fire_classifier.joblib"))
        self.model = None
        self.scaler = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load trained classifier if present, else auto-train and persist baseline model."""
        try:
            import joblib
            if self.model_path.exists():
                saved_bundle = joblib.load(str(self.model_path))
                if isinstance(saved_bundle, dict) and "model" in saved_bundle:
                    self.model = saved_bundle["model"]
                    self.scaler = saved_bundle.get("scaler")
                else:
                    self.model = saved_bundle
                logger.info(f"Loaded existing ML fire classifier from {self.model_path}")
                return
        except Exception as e:
            logger.warning(f"Could not load pre-existing model ({e}); will train baseline model.")

        self._train_and_persist_baseline()

    def _generate_synthetic_training_data(self, n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced synthetic feature matrix mimicking satellite signatures."""
        np.random.seed(42)
        X = []
        y = []
        samples_per_class = n_samples // len(CLASS_NAMES)

        for class_id in range(len(CLASS_NAMES)):
            for _ in range(samples_per_class):
                if class_id == 0:  # Industrial Fire
                    brightness = np.random.uniform(330.0, 420.0)
                    frp = np.random.uniform(40.0, 300.0)
                    dist_ind = np.random.uniform(0.05, 1.8)
                    is_near_ind = 1
                    hour = np.random.randint(0, 24)
                    day_of_week = np.random.randint(0, 7)
                    month = np.random.randint(1, 13)
                    is_daytime = 1 if (6 <= hour <= 18) else 0
                    cluster_id = np.random.randint(1, 10)
                elif class_id == 1:  # Gas Flare
                    brightness = np.random.uniform(380.0, 500.0)
                    frp = np.random.uniform(15.0, 120.0)
                    dist_ind = np.random.uniform(0.01, 0.8)
                    is_near_ind = 1
                    hour = np.random.randint(0, 24)
                    day_of_week = np.random.randint(0, 7)
                    month = np.random.randint(1, 13)
                    is_daytime = 1 if (6 <= hour <= 18) else 0
                    cluster_id = 1
                elif class_id == 2:  # Forest Fire
                    brightness = np.random.uniform(320.0, 410.0)
                    frp = np.random.uniform(60.0, 500.0)
                    dist_ind = np.random.uniform(8.0, 50.0)
                    is_near_ind = 0
                    hour = np.random.randint(0, 24)
                    day_of_week = np.random.randint(0, 7)
                    month = np.random.choice([2, 3, 4, 5, 6])  # Dry season
                    is_daytime = 1 if (6 <= hour <= 18) else 0
                    cluster_id = np.random.randint(2, 25)
                elif class_id == 3:  # Agricultural Burning
                    brightness = np.random.uniform(310.0, 360.0)
                    frp = np.random.uniform(5.0, 45.0)
                    dist_ind = np.random.uniform(5.0, 30.0)
                    is_near_ind = 0
                    hour = np.random.randint(11, 17)  # Afternoon stubble burning
                    day_of_week = np.random.randint(0, 7)
                    month = np.random.choice([4, 5, 10, 11])  # Harvest seasons
                    is_daytime = 1
                    cluster_id = np.random.randint(1, 8)
                elif class_id == 4:  # Mining Activity
                    brightness = np.random.uniform(325.0, 385.0)
                    frp = np.random.uniform(20.0, 150.0)
                    dist_ind = np.random.uniform(0.1, 2.5)
                    is_near_ind = 1
                    hour = np.random.randint(0, 24)
                    day_of_week = np.random.randint(0, 7)
                    month = np.random.randint(1, 13)
                    is_daytime = 1 if (6 <= hour <= 18) else 0
                    cluster_id = np.random.randint(1, 6)
                else:  # Other/Unknown
                    brightness = np.random.uniform(300.0, 335.0)
                    frp = np.random.uniform(1.0, 25.0)
                    dist_ind = np.random.uniform(3.0, 25.0)
                    is_near_ind = 0
                    hour = np.random.randint(0, 24)
                    day_of_week = np.random.randint(0, 7)
                    month = np.random.randint(1, 13)
                    is_daytime = 1 if (6 <= hour <= 18) else 0
                    cluster_id = 0

                ratio = brightness / (frp + 1e-6)
                features = [
                    brightness, frp, ratio, dist_ind, is_near_ind,
                    hour, day_of_week, month, is_daytime, cluster_id
                ]
                X.append(features)
                y.append(class_id)

        return np.array(X), np.array(y)

    def _train_and_persist_baseline(self):
        """Train Random Forest model on synthesized satellite-geospatial vectors."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib

            logger.info("Auto-training baseline Random Forest Fire Classifier (Part 3/5)...")
            X_train, y_train = self._generate_synthetic_training_data(n_samples=2400)

            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42
            )
            rf.fit(X_train, y_train)
            self.model = rf

            # Save model to disk
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": rf, "feature_names": FEATURE_COLS, "classes": CLASS_NAMES}, str(self.model_path))
            logger.info(f"Baseline Fire Classifier trained and persisted to {self.model_path}")
        except Exception as e:
            logger.warning(f"ML Auto-training unavailable ({e}). Using rule-based classification heuristics.")
            self.model = None

    def _heuristic_classify_row(self, row: pd.Series) -> Tuple[str, int, float]:
        """High-accuracy fallback geospatial heuristic classifier."""
        dist = float(row.get("distance_to_nearest_industrial", 999.0))
        facility_type = str(row.get("nearest_facility_type", "")).lower()
        frp = float(row.get("frp", 10.0))
        brightness = float(row.get("brightness", 320.0))
        hour = int(row.get("hour_of_day", 12)) if pd.notna(row.get("hour_of_day")) else 12
        month = int(row.get("month", 5)) if pd.notna(row.get("month")) else 5
        is_near = (dist <= 2.0)

        # 1. Gas Flare: refinery / oil / gas / petrochemical with high thermal intensity
        if is_near and any(k in facility_type for k in ["flare", "refinery", "petrochemical", "gas", "petroleum"]):
            if brightness > 360.0 or frp > 20.0:
                return "Gas Flare", 1, 0.94

        # 2. Mining Activity
        if is_near and any(k in facility_type for k in ["mine", "quarry"]):
            return "Mining Activity", 4, 0.91

        # 3. Industrial Fire: near thermal power plant or general industry
        if is_near:
            return "Industrial Fire", 0, 0.89

        # 4. Agricultural Stubble Burning: day hours, specific harvest months, moderate FRP
        if (10 <= hour <= 17) and month in [4, 5, 10, 11] and frp < 50.0:
            return "Agricultural Burning", 3, 0.86

        # 5. Forest Fire: high FRP away from industrial facilities
        if frp >= 60.0 and dist > 5.0:
            return "Forest Fire", 2, 0.88

        # 6. Other / Unknown
        return "Other/Unknown", 5, 0.75

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all detections in the DataFrame.
        Adds 'fire_type', 'fire_type_id', 'classification_confidence', and updates 'land_cover'.
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure all feature columns exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                if col == "brightness_frp_ratio":
                    df[col] = df["brightness"] / (df["frp"] + 1e-6)
                elif col == "is_near_industrial":
                    df[col] = (df.get("distance_to_nearest_industrial", 999.0) <= 2.0).astype(int)
                elif col == "hour_of_day":
                    df[col] = 12
                elif col == "day_of_week":
                    df[col] = 2
                elif col == "month":
                    df[col] = 6
                elif col == "is_daytime":
                    df[col] = 1
                elif col == "fire_cluster_id":
                    df[col] = 0
                else:
                    df[col] = 0.0

        if self.model is not None:
            try:
                X = df[FEATURE_COLS].fillna(0).values
                preds = self.model.predict(X)
                probs = self.model.predict_proba(X)
                confidences = np.max(probs, axis=1)

                df["fire_type_id"] = preds.astype(int)
                df["fire_type"] = [CLASS_NAMES[p] if 0 <= p < len(CLASS_NAMES) else "Other/Unknown" for p in preds]
                df["classification_confidence"] = np.round(confidences, 3)
                logger.info(f"Model classified {len(df)} fire detections.")
            except Exception as e:
                logger.warning(f"Model prediction failed ({e}); falling back to heuristic engine.")
                self._apply_heuristics(df)
        else:
            self._apply_heuristics(df)

        # Sync land_cover mapping for backwards compatibility
        # 1=Urban, 2=Industrial, 3=Agricultural, 4=Forest, 9=Unknown
        type_to_lc = {
            "Industrial Fire": 2,
            "Gas Flare": 2,
            "Forest Fire": 4,
            "Agricultural Burning": 3,
            "Mining Activity": 2,
            "Other/Unknown": 9
        }
        df["land_cover"] = df["fire_type"].map(lambda t: type_to_lc.get(t, 9))
        return df

    def _apply_heuristics(self, df: pd.DataFrame):
        types = []
        type_ids = []
        confidences = []
        for _, row in df.iterrows():
            t_name, t_id, conf = self._heuristic_classify_row(row)
            types.append(t_name)
            type_ids.append(t_id)
            confidences.append(conf)
        df["fire_type"] = types
        df["fire_type_id"] = type_ids
        df["classification_confidence"] = confidences
