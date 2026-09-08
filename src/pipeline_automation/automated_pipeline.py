import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import geopandas as gpd

from config import settings
from src.preprocessing import FireDataCleaner, FeatureEngineer
from src.pipeline_automation.database import FireMonitoringDatabase
from src.pipeline_automation.classifier import FireClassifier
from src.pipeline_automation.incremental_fetcher import IncrementalDataFetcher
from src.web.map_generator import MapGenerator
from src.web.demo_data import DemoDataGenerator

logger = logging.getLogger(__name__)

class AutomatedPipeline:
    """
    Executes the 5.1 Automated Data Pipeline:
    1. Incremental anomaly fetch
    2. Feature engineering & spatial joins
    3. Machine learning auto-classification
    4. Persistent database update (SQLite)
    5. Dashboard & map refresh
    """

    def __init__(self, db_path: Optional[Path] = None, model_path: Optional[Path] = None):
        self.db = FireMonitoringDatabase(db_path=db_path)
        self.fetcher = IncrementalDataFetcher()
        self.classifier = FireClassifier(model_path=model_path)
        self.cleaner = FireDataCleaner()
        self.engineer = FeatureEngineer()
        self.map_generator = MapGenerator()
        self.demo_gen = DemoDataGenerator()

    def _get_or_load_facilities(self) -> gpd.GeoDataFrame:
        """Load industrial facilities from geojson cache, or generate baseline."""
        facilities_path = settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
        if facilities_path.exists():
            try:
                gdf = gpd.read_file(facilities_path)
                if not gdf.empty and "latitude" in gdf.columns:
                    return gdf
            except Exception as e:
                logger.warning(f"Could not read cached facilities GeoJSON: {e}")

        # Fallback to demo facilities
        logger.info("Generating baseline industrial facility dataset...")
        gdf = self.demo_gen.generate_facilities()
        try:
            settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            gdf.to_file(facilities_path, driver="GeoJSON")
        except Exception:
            pass
        return gdf

    def run_pipeline(self, simulate: bool = False, day_range: int = 1) -> Dict[str, Any]:
        """
        Execute a complete end-to-end incremental pipeline cycle.
        """
        start_time = time.time()
        logger.info("===============================================================")
        logger.info("   STARTING PART 5.1: AUTOMATED DATA PIPELINE CYCLE")
        logger.info("===============================================================")

        try:
            # 1. Fetch Incremental Data
            logger.info("Step 1/5: Fetching incremental satellite thermal detections...")
            existing_ids = self.db.get_existing_detection_ids()
            raw_new_df, total_fetched = self.fetcher.fetch_incremental_data(
                existing_ids=existing_ids,
                simulate=simulate,
                day_range=day_range
            )

            if raw_new_df.empty:
                logger.info("No new fire detections detected since last run.")
                duration = round(time.time() - start_time, 2)
                self.db.log_pipeline_run("SUCCESS", total_fetched, 0, duration, "No new detections")
                self.fetcher.save_state(0)
                stats = self.db.get_statistics()
                return {
                    "status": "SUCCESS",
                    "records_fetched": total_fetched,
                    "records_new": 0,
                    "duration_seconds": duration,
                    "stats": stats
                }

            # 2. Clean & Preprocess
            logger.info(f"Step 2/5: Preprocessing {len(raw_new_df)} new detections & spatial joins...")
            clean_df = self.cleaner.clean_firms_data(raw_new_df)
            facilities_gdf = self._get_or_load_facilities()
            enriched_df = self.engineer.engineer_features(clean_df, facilities_gdf)

            # 3. Auto-Classification
            logger.info("Step 3/5: Running thermal anomalies through AI/ML Classification Model...")
            classified_df = self.classifier.classify_dataframe(enriched_df)

            # 4. Persistent Database Update
            logger.info("Step 4/5: Appending classified detections to persistent SQLite storage...")
            inserted_count = self.db.insert_fire_detections(classified_df)
            self.db.export_to_csv_and_geojson()

            # Record latest datetime for checkpoint
            latest_dt = str(classified_df["datetime"].max()) if "datetime" in classified_df.columns else None
            self.fetcher.save_state(inserted_count, latest_dt)

            # 5. Dashboard Refresh
            logger.info("Step 5/5: Refreshing live dashboard and regenerating geospatial maps...")
            self.refresh_dashboard_maps(facilities_gdf)

            duration = round(time.time() - start_time, 2)
            self.db.log_pipeline_run("SUCCESS", total_fetched, inserted_count, duration, "Incremental run completed")

            stats = self.db.get_statistics()
            logger.info(f"Pipeline cycle completed successfully in {duration}s! New records: {inserted_count}")
            logger.info(f"Current database total: {stats['total_fires']} thermal sources.")

            return {
                "status": "SUCCESS",
                "records_fetched": total_fetched,
                "records_new": inserted_count,
                "duration_seconds": duration,
                "stats": stats
            }

        except Exception as e:
            duration = round(time.time() - start_time, 2)
            logger.exception(f"Pipeline execution failed: {e}")
            self.db.log_pipeline_run("FAILED", 0, 0, duration, str(e))
            return {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": duration
            }

    def refresh_dashboard_maps(self, facilities_gdf: Optional[gpd.GeoDataFrame] = None):
        """
        Regenerate Folium map HTML files in root and docs/ directory (for GitHub Pages),
        and export latest statistics JSON.
        """
        try:
            fires_df = self.db.get_all_fires(limit=3000)
            if fires_df.empty:
                logger.warning("No fires in database; skipping map refresh.")
                return

            if facilities_gdf is None or facilities_gdf.empty:
                facilities_gdf = self._get_or_load_facilities()

            selected_types = list(self.map_generator.FIRE_TYPE_COLORS.keys())
            map_html = self.map_generator.create_dashboard_map(fires_df, facilities_gdf, selected_types)

            # 1. Update root map.html
            root_map = settings.BASE_DIR / "map.html"
            with open(root_map, "w", encoding="utf-8") as f:
                f.write(map_html)
            logger.info(f"Updated {root_map}")

            # 2. Update docs/map.html for GitHub Pages
            docs_map = settings.BASE_DIR / "docs" / "map.html"
            docs_map.parent.mkdir(parents=True, exist_ok=True)
            with open(docs_map, "w", encoding="utf-8") as f:
                f.write(map_html)
            logger.info(f"Updated {docs_map}")

            # 3. Export latest statistics JSON
            stats = self.map_generator.get_fire_statistics(fires_df)
            stats_path = settings.PROCESSED_DATA_DIR / "latest_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Updated {stats_path}")

        except Exception as e:
            logger.error(f"Failed to refresh dashboard maps: {e}")
