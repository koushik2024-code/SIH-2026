import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Set, Dict, Any, List
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config import settings

logger = logging.getLogger(__name__)

class FireMonitoringDatabase:
    """
    Manages persistent SQLite storage for active fire anomalies, industrial facilities,
    and automated pipeline execution audit logs.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or (settings.DATA_DIR / "fire_monitoring.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a sqlite3 connection with WAL mode enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self):
        """Initialize database schema with tables and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Fire Detections Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS fire_detections (
                detection_id TEXT PRIMARY KEY,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                brightness REAL,
                frp REAL,
                brightness_frp_ratio REAL,
                acq_date TEXT,
                acq_time TEXT,
                datetime TEXT,
                satellite TEXT,
                confidence_level TEXT,
                is_daytime INTEGER DEFAULT 1,
                distance_to_industry_km REAL,
                nearest_facility_name TEXT,
                nearest_facility_type TEXT,
                is_near_industrial INTEGER DEFAULT 0,
                fire_type TEXT,
                fire_type_id INTEGER,
                classification_confidence REAL,
                land_cover INTEGER DEFAULT 9,
                fire_cluster_id INTEGER DEFAULT 0,
                ingested_at TEXT NOT NULL
            );
            """)

            # 2. Pipeline Execution Audit Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                records_fetched INTEGER DEFAULT 0,
                records_new INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0.0,
                notes TEXT
            );
            """)

            # 3. Industrial Facilities Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS industrial_facilities (
                facility_id TEXT PRIMARY KEY,
                name TEXT,
                facility_type TEXT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                source TEXT DEFAULT 'OSM',
                updated_at TEXT
            );
            """)

            # Create Indexes for fast querying & spatial slicing
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fires_coords ON fire_detections(latitude, longitude);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fires_date ON fire_detections(acq_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fires_type ON fire_detections(fire_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fires_near_ind ON fire_detections(is_near_industrial);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_time ON pipeline_runs(timestamp);")

            conn.commit()
            logger.debug(f"Database initialized at {self.db_path}")

    def get_existing_detection_ids(self) -> Set[str]:
        """Fetch all existing detection IDs to filter incoming duplicates."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT detection_id FROM fire_detections")
            return {row["detection_id"] for row in cursor.fetchall()}

    def insert_fire_detections(self, df: pd.DataFrame) -> int:
        """
        Insert fire detections into SQLite database.
        Uses INSERT OR IGNORE on detection_id to guarantee idempotency.
        Returns the number of newly inserted records.
        """
        if df.empty:
            return 0

        inserted_count = 0
        now_str = datetime.utcnow().isoformat()

        records = []
        for _, row in df.iterrows():
            detection_id = str(row.get("detection_id", ""))
            if not detection_id:
                continue

            records.append((
                detection_id,
                float(row.get("latitude", 0.0)),
                float(row.get("longitude", 0.0)),
                float(row.get("brightness", 0.0)) if pd.notna(row.get("brightness")) else None,
                float(row.get("frp", 0.0)) if pd.notna(row.get("frp")) else None,
                float(row.get("brightness_frp_ratio", 0.0)) if pd.notna(row.get("brightness_frp_ratio")) else None,
                str(row.get("acq_date", "")) if pd.notna(row.get("acq_date")) else "",
                str(row.get("acq_time", "")) if pd.notna(row.get("acq_time")) else "",
                str(row.get("datetime", "")) if pd.notna(row.get("datetime")) else "",
                str(row.get("satellite", "UNKNOWN")),
                str(row.get("confidence", row.get("confidence_level", "nominal"))),
                int(row.get("is_daytime", 1)) if pd.notna(row.get("is_daytime")) else 1,
                float(row.get("distance_to_nearest_industrial", row.get("distance_to_industry_km", -1.0))),
                str(row.get("nearest_facility_name", "none")),
                str(row.get("nearest_facility_type", "none")),
                int(row.get("is_near_industrial", 0)),
                str(row.get("fire_type", "Other/Unknown")),
                int(row.get("fire_type_id", 5)),
                float(row.get("classification_confidence", 0.85)),
                int(row.get("land_cover", 9)),
                int(row.get("fire_cluster_id", 0)),
                now_str
            ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            initial_changes = conn.total_changes
            cursor.executemany("""
            INSERT OR IGNORE INTO fire_detections (
                detection_id, latitude, longitude, brightness, frp, brightness_frp_ratio,
                acq_date, acq_time, datetime, satellite, confidence_level, is_daytime,
                distance_to_industry_km, nearest_facility_name, nearest_facility_type,
                is_near_industrial, fire_type, fire_type_id, classification_confidence,
                land_cover, fire_cluster_id, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, records)
            conn.commit()
            inserted_count = conn.total_changes - initial_changes

        logger.info(f"Database update: Inserted {inserted_count} new fire detections (Total batch: {len(records)})")
        return inserted_count

    def log_pipeline_run(self, status: str, records_fetched: int, records_new: int, 
                         duration_seconds: float, notes: str = "") -> int:
        """Log pipeline execution results into pipeline_runs table."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO pipeline_runs (timestamp, status, records_fetched, records_new, duration_seconds, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, status, records_fetched, records_new, duration_seconds, notes))
            conn.commit()
            return cursor.lastrowid

    def get_all_fires(self, limit: Optional[int] = None, fire_type: Optional[str] = None) -> pd.DataFrame:
        """Retrieve fire records from SQLite as a pandas DataFrame."""
        query = "SELECT * FROM fire_detections"
        params = []
        if fire_type:
            query += " WHERE fire_type = ?"
            params.append(fire_type)
        query += " ORDER BY datetime DESC, ingested_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            # Map column names to maintain backward compatibility with visualization
            if "distance_to_industry_km" in df.columns and "distance_to_nearest_industrial" not in df.columns:
                df["distance_to_nearest_industrial"] = df["distance_to_industry_km"]
            if "confidence_level" in df.columns and "confidence" not in df.columns:
                df["confidence"] = df["confidence_level"]
        return df

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics from persistent SQLite storage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total, AVG(brightness) as avg_bright, AVG(frp) as avg_frp FROM fire_detections")
            agg = cursor.fetchone()
            total_fires = agg["total"] or 0
            avg_brightness = round(agg["avg_bright"] or 0.0, 2)
            avg_frp = round(agg["avg_frp"] or 0.0, 2)

            cursor.execute("SELECT fire_type, COUNT(*) as cnt FROM fire_detections GROUP BY fire_type")
            fire_type_distribution = {row["fire_type"]: row["cnt"] for row in cursor.fetchall()}

            cursor.execute("SELECT is_daytime, COUNT(*) as cnt FROM fire_detections GROUP BY is_daytime")
            daynight_counts = {row["is_daytime"]: row["cnt"] for row in cursor.fetchall()}
            day_count = daynight_counts.get(1, 0)
            night_count = daynight_counts.get(0, 0)

            cursor.execute("SELECT COUNT(*) FROM fire_detections WHERE is_near_industrial = 1")
            near_industrial = cursor.fetchone()[0]

            cursor.execute("SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT 1")
            last_run = cursor.fetchone()
            last_run_dict = dict(last_run) if last_run else None

        return {
            "total_fires": total_fires,
            "avg_brightness": avg_brightness,
            "avg_frp": avg_frp,
            "fire_type_distribution": fire_type_distribution,
            "day_fires": day_count,
            "night_fires": night_count,
            "near_industrial_count": near_industrial,
            "last_pipeline_run": last_run_dict,
        }

    def get_recent_pipeline_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent pipeline execution runs for audit and monitoring."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def export_to_csv_and_geojson(self, csv_path: Optional[Path] = None, geojson_path: Optional[Path] = None):
        """
        Synchronize SQLite records into CSV and GeoJSON for backward compatibility
        with Part 1 and dashboard components.
        """
        df = self.get_all_fires()
        if df.empty:
            logger.warning("Database empty; skipping export.")
            return

        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_target = Path(csv_path or (settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"))
        df.to_csv(csv_target, index=False)
        logger.info(f"Exported {len(df)} records to {csv_target}")

        try:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )
            geojson_target = Path(geojson_path or (settings.PROCESSED_DATA_DIR / "latest_detections.geojson"))
            gdf.to_file(geojson_target, driver="GeoJSON")
            logger.info(f"Exported GeoJSON to {geojson_target}")
        except Exception as e:
            logger.warning(f"GeoJSON export skipped or failed: {e}")
