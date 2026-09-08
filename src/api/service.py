import logging
import math
from typing import List, Optional, Dict, Any
import pandas as pd

from config import settings
from src.pipeline_automation.database import FireMonitoringDatabase
from src.pipeline_automation.classifier import FireClassifier
from src.web.demo_data import DemoDataGenerator
from src.api.models import (
    FireRecord,
    FireListResponse,
    FacilityRecord,
    FacilityListResponse,
    HotspotRecord,
    HotspotListResponse,
    StatsResponse,
    ClassifyFeatureInput,
    ClassifyResult,
    ClassifyBatchResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

class APIService:
    """
    Business logic and data access service powering the FastAPI REST endpoints.
    """

    def __init__(self, db: Optional[FireMonitoringDatabase] = None):
        self.db = db or FireMonitoringDatabase()
        self.classifier = FireClassifier()
        self.demo_gen = DemoDataGenerator()
        self._ensure_initial_facilities()

    def _ensure_initial_facilities(self):
        """Seed industrial facilities table if currently empty."""
        try:
            existing = self.db.get_facilities(limit=1)
            if not existing:
                demo_facs = self.demo_gen.generate_facilities()
                if hasattr(demo_facs, "iterrows"):
                    records = []
                    for idx, row in demo_facs.iterrows():
                        records.append({
                            "facility_id": f"FAC-{idx+1:04d}",
                            "name": str(row.get("name", "Unknown Facility")),
                            "facility_type": str(row.get("type", "industrial")),
                            "latitude": float(row.geometry.y) if hasattr(row, "geometry") else float(row.get("lat", 0.0)),
                            "longitude": float(row.geometry.x) if hasattr(row, "geometry") else float(row.get("lon", 0.0)),
                            "source": "OSM"
                        })
                    self.db.insert_facilities(records)
                    logger.info(f"Seeded {len(records)} industrial facilities into SQLite database.")
        except Exception as e:
            logger.warning(f"Could not check/seed facilities in database: {e}")

    def get_fires(
        self,
        limit: int = 50,
        offset: int = 0,
        fire_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        is_near_industrial: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> FireListResponse:
        """Fetch fire detections with comprehensive filtering and pagination."""
        df = self.db.get_all_fires(limit=3000)

        # Fallback to demo synthesizer if database is empty
        if df.empty:
            df = self.demo_gen.generate_fire_data(n_fires=250, days_back=3)

        if "detection_id" not in df.columns:
            df["detection_id"] = [f"DET-{i+1:05d}" for i in range(len(df))]

        if fire_type and "fire_type" in df.columns:
            df = df[df["fire_type"].str.lower() == fire_type.strip().lower()]

        if is_near_industrial is not None and "is_near_industrial" in df.columns:
            df = df[df["is_near_industrial"] == int(is_near_industrial)]

        if start_date and "acq_date" in df.columns:
            df = df[df["acq_date"].astype(str) >= start_date.strip()]
        if end_date and "acq_date" in df.columns:
            df = df[df["acq_date"].astype(str) <= end_date.strip()]

        if min_confidence is not None and "confidence" in df.columns:
            def conf_to_float(v):
                try:
                    return float(v)
                except Exception:
                    s = str(v).lower()
                    if "high" in s or s == "h":
                        return 85.0
                    elif "nominal" in s or "med" in s or s == "n":
                        return 60.0
                    return 30.0
            confs = df["confidence"].apply(conf_to_float)
            df = df[confs >= float(min_confidence)]

        total = len(df)
        paginated_df = df.iloc[offset : offset + limit]

        records: List[FireRecord] = []
        for _, row in paginated_df.iterrows():
            det_id = str(row.get("detection_id", ""))
            records.append(FireRecord(
                detection_id=det_id,
                latitude=float(row.get("latitude", 0.0)),
                longitude=float(row.get("longitude", 0.0)),
                brightness=float(row.get("brightness", 0.0)) if pd.notna(row.get("brightness")) else None,
                frp=float(row.get("frp", 0.0)) if pd.notna(row.get("frp")) else None,
                brightness_frp_ratio=float(row.get("brightness_frp_ratio", 0.0)) if pd.notna(row.get("brightness_frp_ratio")) else None,
                acq_date=str(row.get("acq_date", "")),
                acq_time=str(row.get("acq_time", "")),
                datetime=str(row.get("datetime", "")),
                satellite=str(row.get("satellite", "UNKNOWN")),
                confidence=row.get("confidence", "nominal"),
                is_daytime=int(row.get("is_daytime", 1)) if pd.notna(row.get("is_daytime")) else 1,
                distance_to_nearest_industrial=float(row.get("distance_to_nearest_industrial", -1.0)) if pd.notna(row.get("distance_to_nearest_industrial")) else None,
                nearest_facility_name=str(row.get("nearest_facility_name", "none")),
                nearest_facility_type=str(row.get("nearest_facility_type", "none")),
                is_near_industrial=int(row.get("is_near_industrial", 0)) if pd.notna(row.get("is_near_industrial")) else 0,
                fire_type=str(row.get("fire_type", "Other/Unknown")),
                fire_type_id=int(row.get("fire_type_id", 5)) if pd.notna(row.get("fire_type_id")) else 5,
                classification_confidence=float(row.get("classification_confidence", 0.85)) if pd.notna(row.get("classification_confidence")) else 0.85,
                land_cover=int(row.get("land_cover", 9)) if pd.notna(row.get("land_cover")) else 9,
                fire_cluster_id=int(row.get("fire_cluster_id", 0)) if pd.notna(row.get("fire_cluster_id")) else 0,
                ingested_at=str(row.get("ingested_at", "")) if pd.notna(row.get("ingested_at")) else None
            ))

        return FireListResponse(
            total=total,
            count=len(records),
            limit=limit,
            offset=offset,
            data=records
        )

    def get_fire_by_id(self, detection_id: str) -> Optional[FireRecord]:
        """Retrieve a specific fire record by primary detection ID."""
        row_dict = self.db.get_fire_by_id(detection_id)
        if not row_dict:
            # Check demo generator fallback
            df = self.demo_gen.generate_fire_data(n_fires=250, days_back=3)
            if "detection_id" not in df.columns:
                df["detection_id"] = [f"DET-{i+1:05d}" for i in range(len(df))]
            matches = df[df["detection_id"] == detection_id]
            if not matches.empty:
                row_dict = matches.iloc[0].to_dict()

        if not row_dict:
            return None

        return FireRecord(
            detection_id=str(row_dict.get("detection_id", "")),
            latitude=float(row_dict.get("latitude", 0.0)),
            longitude=float(row_dict.get("longitude", 0.0)),
            brightness=float(row_dict.get("brightness", 0.0)) if pd.notna(row_dict.get("brightness")) else None,
            frp=float(row_dict.get("frp", 0.0)) if pd.notna(row_dict.get("frp")) else None,
            brightness_frp_ratio=float(row_dict.get("brightness_frp_ratio", 0.0)) if pd.notna(row_dict.get("brightness_frp_ratio")) else None,
            acq_date=str(row_dict.get("acq_date", "")),
            acq_time=str(row_dict.get("acq_time", "")),
            datetime=str(row_dict.get("datetime", "")),
            satellite=str(row_dict.get("satellite", "UNKNOWN")),
            confidence=row_dict.get("confidence", "nominal"),
            is_daytime=int(row_dict.get("is_daytime", 1)) if pd.notna(row_dict.get("is_daytime")) else 1,
            distance_to_nearest_industrial=float(row_dict.get("distance_to_nearest_industrial", -1.0)) if pd.notna(row_dict.get("distance_to_nearest_industrial")) else None,
            nearest_facility_name=str(row_dict.get("nearest_facility_name", "none")),
            nearest_facility_type=str(row_dict.get("nearest_facility_type", "none")),
            is_near_industrial=int(row_dict.get("is_near_industrial", 0)) if pd.notna(row_dict.get("is_near_industrial")) else 0,
            fire_type=str(row_dict.get("fire_type", "Other/Unknown")),
            fire_type_id=int(row_dict.get("fire_type_id", 5)) if pd.notna(row_dict.get("fire_type_id")) else 5,
            classification_confidence=float(row_dict.get("classification_confidence", 0.85)) if pd.notna(row_dict.get("classification_confidence")) else 0.85,
            land_cover=int(row_dict.get("land_cover", 9)) if pd.notna(row_dict.get("land_cover")) else 9,
            fire_cluster_id=int(row_dict.get("fire_cluster_id", 0)) if pd.notna(row_dict.get("fire_cluster_id")) else 0,
            ingested_at=str(row_dict.get("ingested_at", "")) if pd.notna(row_dict.get("ingested_at")) else None
        )

    def get_facilities(self, facility_type: Optional[str] = None, limit: int = 100) -> FacilityListResponse:
        """Retrieve registered industrial facilities."""
        facilities = self.db.get_facilities(facility_type=facility_type, limit=limit)
        if not facilities:
            self._ensure_initial_facilities()
            facilities = self.db.get_facilities(facility_type=facility_type, limit=limit)

        records: List[FacilityRecord] = []
        for f in facilities:
            records.append(FacilityRecord(
                facility_id=str(f.get("facility_id", "")),
                name=str(f.get("name", "Unknown")),
                facility_type=str(f.get("facility_type", "industrial")),
                latitude=float(f.get("latitude", 0.0)),
                longitude=float(f.get("longitude", 0.0)),
                source=str(f.get("source", "OSM"))
            ))

        return FacilityListResponse(
            total=len(records),
            data=records
        )

    def get_hotspots(self, min_frp: float = 20.0, limit: int = 50) -> HotspotListResponse:
        """
        Extract and rank high-intensity thermal anomaly hotspots.
        Scores each hotspot based on FRP, temperature brightness, and confidence.
        """
        df = self.db.get_all_fires(limit=3000)
        if df.empty:
            df = self.demo_gen.generate_fire_data(n_fires=250, days_back=3)

        if df.empty:
            return HotspotListResponse(total=0, data=[])

        if "detection_id" not in df.columns:
            df["detection_id"] = [f"DET-{i+1:05d}" for i in range(len(df))]

        # Filter candidates by minimum FRP or high brightness
        candidates = df[(df["frp"] >= min_frp) | (df["brightness"] >= 350.0)].copy()
        if candidates.empty:
            candidates = df.copy()

        hotspots: List[HotspotRecord] = []
        for _, row in candidates.iterrows():
            frp_val = float(row.get("frp", 10.0)) if pd.notna(row.get("frp")) else 10.0
            bright_val = float(row.get("brightness", 320.0)) if pd.notna(row.get("brightness")) else 320.0
            conf_val = row.get("confidence", 80.0)
            
            try:
                conf_num = float(conf_val)
            except Exception:
                conf_num = 85.0 if str(conf_val).lower() in ["h", "high"] else 60.0

            # Composite thermal intensity score
            score = round(frp_val * (bright_val / 300.0) * (conf_num / 100.0), 2)

            hotspots.append(HotspotRecord(
                detection_id=str(row.get("detection_id", "")),
                latitude=float(row.get("latitude", 0.0)),
                longitude=float(row.get("longitude", 0.0)),
                brightness=round(bright_val, 2),
                frp=round(frp_val, 2),
                confidence=conf_val,
                intensity_score=score,
                fire_type=str(row.get("fire_type", "Other/Unknown")),
                facility_nearby=str(row.get("nearest_facility_name", "none")),
                distance_km=round(float(row.get("distance_to_nearest_industrial", -1.0)), 2) if pd.notna(row.get("distance_to_nearest_industrial")) else None,
                cluster_id=int(row.get("fire_cluster_id", 0)) if pd.notna(row.get("fire_cluster_id")) else 0
            ))

        # Sort descending by thermal intensity score
        hotspots.sort(key=lambda h: h.intensity_score, reverse=True)
        top_hotspots = hotspots[:limit]

        return HotspotListResponse(
            total=len(top_hotspots),
            data=top_hotspots
        )

    def get_statistics(self) -> StatsResponse:
        """Compute system-wide thermal and operational statistics."""
        db_stats = self.db.get_statistics()
        total_fires = db_stats.get("total_fires", 0)

        # Fallback if DB is not populated yet
        if total_fires == 0:
            df = self.demo_gen.generate_fire_data(n_fires=250, days_back=3)
            total_fires = len(df)
            avg_b = round(df["brightness"].mean(), 2) if "brightness" in df.columns else 340.0
            avg_f = round(df["frp"].mean(), 2) if "frp" in df.columns else 25.0
            dist = df["fire_type"].value_counts().to_dict() if "fire_type" in df.columns else {}
            day_c = int((df["is_daytime"] == 1).sum()) if "is_daytime" in df.columns else 150
            night_c = total_fires - day_c
            near_ind = int((df["is_near_industrial"] == 1).sum()) if "is_near_industrial" in df.columns else 40
            crit_hotspots = int((df["frp"] >= 40.0).sum()) if "frp" in df.columns else 15
            last_run = None
        else:
            avg_b = db_stats.get("avg_brightness", 0.0)
            avg_f = db_stats.get("avg_frp", 0.0)
            dist = db_stats.get("fire_type_distribution", {})
            day_c = db_stats.get("day_fires", 0)
            night_c = db_stats.get("night_fires", 0)
            near_ind = db_stats.get("near_industrial_count", 0)
            crit_hotspots = len(self.get_hotspots(min_frp=50.0, limit=1000).data)
            last_run = db_stats.get("last_pipeline_run")

        return StatsResponse(
            total_fires=total_fires,
            avg_brightness=float(avg_b),
            avg_frp=float(avg_f),
            fire_type_distribution=dist,
            day_fires=day_c,
            night_fires=night_c,
            near_industrial_count=near_ind,
            critical_hotspots=crit_hotspots,
            last_pipeline_run=last_run
        )

    def classify_records(self, records: List[ClassifyFeatureInput]) -> ClassifyBatchResponse:
        """
        Classify input fire observations into one of 6 classes:
        Industrial Fire, Gas Flare, Forest Fire, Agricultural Burning, Mining Activity, Other/Unknown.
        """
        if not records:
            return ClassifyBatchResponse(total_classified=0, results=[])

        # Convert records to DataFrame
        data = []
        facilities = self.db.get_facilities(limit=500)

        for rec in records:
            # Compute distance to nearest registered facility if not provided
            dist = rec.distance_to_nearest_industrial
            nearest_name = "none"
            nearest_type = rec.nearest_facility_type or "unknown"

            if dist is None and facilities:
                min_d = 999.0
                closest_fac = None
                for fac in facilities:
                    d = haversine(rec.latitude, rec.longitude, fac["latitude"], fac["longitude"])
                    if d < min_d:
                        min_d = d
                        closest_fac = fac
                dist = round(min_d, 3)
                if closest_fac:
                    nearest_name = closest_fac["name"]
                    nearest_type = closest_fac["facility_type"]
            elif dist is None:
                dist = 50.0

            data.append({
                "latitude": rec.latitude,
                "longitude": rec.longitude,
                "brightness": rec.brightness,
                "frp": rec.frp,
                "confidence": rec.confidence,
                "distance_to_nearest_industrial": dist,
                "nearest_facility_name": nearest_name,
                "nearest_facility_type": nearest_type,
                "acq_date": rec.acq_date or "2026-06-01",
                "acq_time": rec.acq_time or "1200",
                "satellite": rec.satellite or "VIIRS",
            })

        df = pd.DataFrame(data)
        classified_df = self.classifier.classify_dataframe(df)

        results: List[ClassifyResult] = []
        for _, row in classified_df.iterrows():
            results.append(ClassifyResult(
                fire_type=str(row.get("fire_type", "Other/Unknown")),
                fire_type_id=int(row.get("fire_type_id", 5)),
                classification_confidence=round(float(row.get("classification_confidence", 0.85)), 3),
                is_near_industrial=int(row.get("is_near_industrial", 0)),
                distance_to_nearest_industrial=round(float(row.get("distance_to_nearest_industrial", -1.0)), 2),
                nearest_facility_name=str(row.get("nearest_facility_name", "none")),
                nearest_facility_type=str(row.get("nearest_facility_type", "none")),
            ))

        return ClassifyBatchResponse(
            total_classified=len(results),
            results=results
        )

    def get_health(self) -> HealthResponse:
        """Verify database accessibility and model readiness."""
        db_connected = False
        fire_count = 0
        facility_count = 0

        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM fire_detections")
                fire_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM industrial_facilities")
                facility_count = cursor.fetchone()[0]
                db_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")

        model_loaded = (self.classifier.model is not None)

        return HealthResponse(
            status="ok" if db_connected else "degraded",
            version="1.0.0",
            database_connected=db_connected,
            total_fires_in_db=fire_count,
            total_facilities_in_db=facility_count,
            ml_classifier_loaded=model_loaded
        )

    def _get_analysis_data(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Fetch fire dataframe and facilities for reporting."""
        df = self.db.get_all_fires(limit=5000)
        if df.empty:
            df = self.demo_gen.generate_fire_data(n_fires=300, days_back=90)
        facilities = self.db.get_facilities(limit=200)
        if not facilities:
            self._ensure_initial_facilities()
            facilities = self.db.get_facilities(limit=200)
        return df, facilities

    def get_monthly_report(self) -> List[Dict[str, Any]]:
        """Compute monthly fire metrics."""
        from src.reporting.report_engine import ReportEngine
        engine = ReportEngine()
        df, _ = self._get_analysis_data()
        return engine.analyzer.compute_monthly_report(df)

    def get_quarterly_report(self) -> List[Dict[str, Any]]:
        """Compute quarterly fire metrics."""
        from src.reporting.report_engine import ReportEngine
        engine = ReportEngine()
        df, _ = self._get_analysis_data()
        return engine.analyzer.compute_quarterly_report(df)

    def get_regional_trends(self) -> List[Dict[str, Any]]:
        """Compute regional surveillance trends."""
        from src.reporting.report_engine import ReportEngine
        engine = ReportEngine()
        df, _ = self._get_analysis_data()
        return engine.analyzer.compute_regional_trends(df)

    def get_facility_risks(self) -> List[Dict[str, Any]]:
        """Compute facility longitudinal risk rankings."""
        from src.reporting.report_engine import ReportEngine
        engine = ReportEngine()
        df, facilities = self._get_analysis_data()
        return engine.analyzer.compute_facility_risk_scores(df, facilities)

    def get_full_report_bundle(self) -> Dict[str, Any]:
        """Generate and retrieve full multi-format report bundle."""
        from src.reporting.report_engine import ReportEngine
        engine = ReportEngine()
        df, facilities = self._get_analysis_data()
        return engine.run_full_analysis(df, facilities)

