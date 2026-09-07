import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Set, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

from config import settings
from src.data_ingestion import FIRMSClient

logger = logging.getLogger(__name__)

class IncrementalDataFetcher:
    """
    Handles incremental thermal anomaly acquisition from NASA FIRMS.
    Maintains run checkpoints, calculates deterministic SHA-256 detection signatures,
    and deduplicates records against persistent storage to only fetch new anomalies.
    """

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = Path(state_file or (settings.DATA_DIR / "pipeline_state.json"))
        self.firms_client = FIRMSClient()
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state checkpoint from disk or initialize default."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read state file ({e}); initializing fresh state.")

        return {
            "last_run_timestamp": None,
            "last_detection_datetime": None,
            "total_detections_ingested": 0,
            "total_runs": 0,
            "status": "INITIALIZED"
        }

    def save_state(self, new_detections_count: int, latest_datetime_str: Optional[str] = None):
        """Save updated run metrics to the state file."""
        self.state["last_run_timestamp"] = datetime.utcnow().isoformat()
        if latest_datetime_str:
            self.state["last_detection_datetime"] = latest_datetime_str
        self.state["total_detections_ingested"] = self.state.get("total_detections_ingested", 0) + new_detections_count
        self.state["total_runs"] = self.state.get("total_runs", 0) + 1
        self.state["status"] = "SUCCESS"

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Pipeline state updated: {self.state}")
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")

    @staticmethod
    def generate_detection_id(lat: float, lon: float, acq_date: str, acq_time: str, satellite: str) -> str:
        """Create a deterministic unique hash signature for a fire detection."""
        norm_lat = f"{float(lat):.4f}"
        norm_lon = f"{float(lon):.4f}"
        norm_date = str(acq_date).strip()
        norm_time = str(acq_time).strip().zfill(4)
        norm_sat = str(satellite).strip().upper()

        key = f"{norm_lat}_{norm_lon}_{norm_date}_{norm_time}_{norm_sat}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def fetch_incremental_data(self, existing_ids: Set[str], 
                               simulate: bool = False,
                               day_range: int = 1) -> Tuple[pd.DataFrame, int]:
        """
        Fetch detection batch and filter out any records that already exist in persistent storage.
        Returns (new_detections_df, total_fetched_count).
        """
        raw_df = pd.DataFrame()

        # Try live NASA FIRMS fetch if not forced to simulate and API key exists
        if not simulate and settings.FIRMS_MAP_KEY:
            try:
                logger.info(f"Connecting to NASA FIRMS API (day_range={day_range})...")
                raw_df = self.firms_client.fetch_all_sources(day_range=day_range)
            except Exception as e:
                logger.warning(f"FIRMS API fetch failed ({e}).")

        # Fallback / simulation if no API key or empty response
        if raw_df.empty:
            if simulate or not settings.FIRMS_MAP_KEY:
                logger.info("Using simulated incremental satellite pass for demonstration...")
                raw_df = self._generate_simulated_pass()
            else:
                logger.warning("No fire data fetched from NASA FIRMS.")
                return pd.DataFrame(), 0

        total_fetched = len(raw_df)
        if total_fetched == 0:
            return pd.DataFrame(), 0

        # Calculate deterministic detection_ids
        detection_ids = []
        for _, row in raw_df.iterrows():
            did = self.generate_detection_id(
                lat=row["latitude"],
                lon=row["longitude"],
                acq_date=row.get("acq_date", datetime.utcnow().strftime("%Y-%m-%d")),
                acq_time=row.get("acq_time", "1200"),
                satellite=row.get("satellite", row.get("source", "NOAA20"))
            )
            detection_ids.append(did)

        raw_df["detection_id"] = detection_ids

        # Filter out existing IDs
        new_df = raw_df[~raw_df["detection_id"].isin(existing_ids)].copy()
        logger.info(f"Incremental fetch: {len(new_df)} new detections found out of {total_fetched} fetched.")
        return new_df, total_fetched

    def _generate_simulated_pass(self) -> pd.DataFrame:
        """
        Simulate a realistic 6-hour satellite pass with anomalies across India.
        Provides realistic thermal signatures for industrial clusters and rural zones.
        """
        np.random.seed(int(datetime.utcnow().timestamp()) % 10000)
        
        # Strategic industrial & active thermal locations across India
        anchor_locations = [
            {"name": "Jamnagar Refinery Complex", "lat": 22.4707, "lon": 70.0577, "type": "oil_refinery", "base_bright": 395.0, "base_frp": 110.0},
            {"name": "Singrauli Thermal Power Belt", "lat": 24.1997, "lon": 82.6645, "type": "thermal_power_plant", "base_bright": 365.0, "base_frp": 85.0},
            {"name": "Korba Coal Mining Belt", "lat": 22.3595, "lon": 82.7501, "type": "mining", "base_bright": 355.0, "base_frp": 65.0},
            {"name": "Angul Steel Plant Cluster", "lat": 20.8444, "lon": 85.1511, "type": "steel_plant", "base_bright": 380.0, "base_frp": 95.0},
            {"name": "Visakhapatnam Petrochemical Zone", "lat": 17.6868, "lon": 83.2185, "type": "petrochemical", "base_bright": 370.0, "base_frp": 70.0},
            {"name": "Punjab Agricultural Stubble Plain", "lat": 30.7333, "lon": 75.8500, "type": "agricultural", "base_bright": 325.0, "base_frp": 25.0},
            {"name": "Western Ghats Forest Zone", "lat": 14.5000, "lon": 74.8000, "type": "forest", "base_bright": 345.0, "base_frp": 130.0},
            {"name": "Mumbai Trombay Industrial Area", "lat": 19.0160, "lon": 72.9000, "type": "refinery", "base_bright": 385.0, "base_frp": 75.0}
        ]

        now = datetime.utcnow()
        current_date_str = now.strftime("%Y-%m-%d")
        current_time_str = now.strftime("%H%M")

        records = []
        n_points = np.random.randint(20, 35)

        for _ in range(n_points):
            anchor = anchor_locations[np.random.randint(0, len(anchor_locations))]
            
            # Scatter within 0.05 - 0.25 degrees (~5 to 25 km)
            dlat = np.random.normal(0, 0.08)
            dlon = np.random.normal(0, 0.08)
            lat = round(anchor["lat"] + dlat, 4)
            lon = round(anchor["lon"] + dlon, 4)

            brightness = round(float(anchor["base_bright"] + np.random.normal(0, 15)), 2)
            frp = max(5.0, round(float(anchor["base_frp"] + np.random.normal(0, 20)), 2))
            sat = np.random.choice(["VIIRS_NOAA20", "VIIRS_SNPP", "MODIS_Aqua"])
            confidence = np.random.choice(["nominal", "high", "high"])
            daynight = "D" if (6 <= now.hour <= 18) else "N"

            records.append({
                "latitude": lat,
                "longitude": lon,
                "brightness": brightness,
                "frp": frp,
                "acq_date": current_date_str,
                "acq_time": current_time_str,
                "satellite": sat,
                "confidence": confidence,
                "daynight": daynight,
                "source": sat
            })

        return pd.DataFrame(records)
