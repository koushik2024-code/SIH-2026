import argparse
import logging
import sys

def setup_logging():
    """Setup basic logging for the entry point."""
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def print_banner():
    """Print project banner."""
    banner = """
    ===============================================================
       INDUSTRIAL FIRE & PERSISTENT THERMAL SOURCE DETECTION
                     SIH 2026 | NTRO Challenge
         NASA FIRMS  |  OpenStreetMap  |  Satellite Data
    ===============================================================
    """
    try:
        print(banner)
    except Exception:
        pass

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Industrial Fire Detection System CLI")
    parser.add_argument("--part", type=str, choices=['1', '2', '2.2', '2.3', '2.4', '3', '4', '5', 'web'], default='web',
                        help="Which part to run (1-5, 2.2, 2.3, 2.4) or 'web' for dashboard")
    parser.add_argument("--bbox", type=str, help="Bounding box as W,S,E,N", default=None)
    parser.add_argument("--days", type=int, help="Number of days of data to fetch", default=2)
    parser.add_argument("--port", type=int, help="Web server port", default=5000)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        if args.part == '1':
            from src.pipeline import DataPipeline
            logger.info("Starting Part 1: Data Ingestion & Preprocessing Pipeline")
            pipeline = DataPipeline()
            pipeline.run_full_pipeline()
            logger.info("Part 1 execution completed successfully.")
        elif args.part == '2':
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.analysis.proximity_engine import ProximityAnalysisEngine
            
            logger.info("Starting Part 2: Proximity Analysis Engine")
            
            fires_path = settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            facilities_path = settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
            
            if not fires_path.exists() or not facilities_path.exists():
                logger.error("Missing input data for Part 2. Run Part 1 first.")
                sys.exit(1)
                
            logger.info("Loading datasets...")
            fires_df = pd.read_csv(fires_path)
            
            if fires_df.empty:
                logger.error("Fire dataset is empty. Cannot perform proximity analysis.")
                sys.exit(1)
                
            # Convert fires to GeoDataFrame
            fires_gdf = gpd.GeoDataFrame(
                fires_df, 
                geometry=gpd.points_from_xy(fires_df.longitude, fires_df.latitude),
                crs="EPSG:4326"
            )
            
            facilities_gdf = gpd.read_file(facilities_path)
            
            # Find facility ID column
            facility_id_col = "id" if "id" in facilities_gdf.columns else "osm_id"
            if facility_id_col not in facilities_gdf.columns:
                facilities_gdf['facility_id'] = facilities_gdf.index
                facility_id_col = 'facility_id'
                
            # Create a mock fire_id if not present
            if "fire_id" not in fires_gdf.columns:
                fires_gdf["fire_id"] = [f"fire_{i}" for i in range(len(fires_gdf))]
                
            engine = ProximityAnalysisEngine()
            final_ranked_gdf, buffers_gdf = engine.run_analysis(
                fires_gdf, 
                facilities_gdf, 
                fire_id_col="fire_id", 
                facility_id_col=facility_id_col
            )
            
            out_fires = settings.PROCESSED_DATA_DIR / "proximity_scored_fires.csv"
            out_buffers = settings.PROCESSED_DATA_DIR / "facility_buffers.geojson"
            
            # Save results
            pd.DataFrame(final_ranked_gdf.drop(columns=['geometry'])).to_csv(out_fires, index=False)
            buffers_gdf.to_file(out_buffers, driver="GeoJSON")
            
            logger.info(f"Saved scored fires to {out_fires}")
            logger.info(f"Saved facility buffers to {out_buffers}")
            logger.info("Part 2 execution completed successfully.")
        elif args.part == '2.2':
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.analysis.hotspot_engine import HotspotEngine
            
            logger.info("Starting Part 2.2: Hotspot Detection Engine")
            
            fires_path = settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            
            if not fires_path.exists():
                logger.error("Missing input data for Part 2.2. Run Part 1 first.")
                sys.exit(1)
                
            logger.info("Loading dataset...")
            fires_df = pd.read_csv(fires_path)
            
            if fires_df.empty:
                logger.error("Fire dataset is empty. Cannot perform hotspot analysis.")
                sys.exit(1)
                
            # Convert fires to GeoDataFrame
            fires_gdf = gpd.GeoDataFrame(
                fires_df, 
                geometry=gpd.points_from_xy(fires_df.longitude, fires_df.latitude),
                crs="EPSG:4326"
            )
            
            engine = HotspotEngine()
            hotspots_gdf, kde_surface = engine.run_pipeline(fires_gdf)
            
            out_hotspots = settings.PROCESSED_DATA_DIR / "hotspots_analyzed.csv"
            out_kde = settings.PROCESSED_DATA_DIR / "hotspots_kde_surface.geojson"
            
            pd.DataFrame(hotspots_gdf.drop(columns=['geometry'])).to_csv(out_hotspots, index=False)
            
            if not kde_surface.empty:
                kde_surface.to_file(out_kde, driver="GeoJSON")
            
            logger.info(f"Saved analyzed hotspots to {out_hotspots}")
            logger.info(f"Saved KDE heat surface to {out_kde}")
            logger.info("Part 2.2 execution completed successfully.")
        elif args.part == '2.3':
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.analysis.temporal_clustering import TemporalClusteringEngine
            
            logger.info("Starting Part 2.3: Temporal Clustering Engine")
            
            fires_path = settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            
            if not fires_path.exists():
                logger.error("Missing input data for Part 2.3. Run Part 1 first.")
                sys.exit(1)
                
            logger.info("Loading dataset...")
            fires_df = pd.read_csv(fires_path)
            
            if fires_df.empty:
                logger.error("Fire dataset is empty. Cannot perform temporal clustering.")
                sys.exit(1)
                
            # Convert fires to GeoDataFrame
            fires_gdf = gpd.GeoDataFrame(
                fires_df, 
                geometry=gpd.points_from_xy(fires_df.longitude, fires_df.latitude),
                crs="EPSG:4326"
            )
            
            engine = TemporalClusteringEngine()
            clustered_fires_gdf, cluster_hulls_gdf = engine.run_pipeline(fires_gdf)
            
            out_fires = settings.PROCESSED_DATA_DIR / "temporal_clustered_fires.csv"
            out_hulls = settings.PROCESSED_DATA_DIR / "temporal_cluster_hulls.geojson"
            
            # Save results
            # Note: We need to drop geometry and convert datetime to string for CSV saving
            df_to_save = pd.DataFrame(clustered_fires_gdf.drop(columns=['geometry']))
            if 'datetime' in df_to_save.columns:
                df_to_save['datetime'] = df_to_save['datetime'].astype(str)
            df_to_save.to_csv(out_fires, index=False)
            
            if not cluster_hulls_gdf.empty:
                # Convert datetime to string for GeoJSON serialization
                hulls_to_save = cluster_hulls_gdf.copy()
                if 'first_detection' in hulls_to_save.columns:
                    hulls_to_save['first_detection'] = hulls_to_save['first_detection'].astype(str)
                if 'last_detection' in hulls_to_save.columns:
                    hulls_to_save['last_detection'] = hulls_to_save['last_detection'].astype(str)
                hulls_to_save.to_file(out_hulls, driver="GeoJSON")
            
            logger.info(f"Saved clustered fires to {out_fires}")
            logger.info(f"Saved cluster hulls to {out_hulls}")
            logger.info("Part 2.3 execution completed successfully.")
        elif args.part == '2.4':
            import pandas as pd
            import geopandas as gpd
            import json
            from config import settings
            from src.analysis.spatial_statistics import SpatialStatisticsEngine
            
            logger.info("Starting Part 2.4: Spatial Statistics & Data Enrichment")
            
            fires_path = settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            
            if not fires_path.exists():
                logger.error("Missing input data for Part 2.4. Run Part 1 first.")
                sys.exit(1)
                
            logger.info("Loading dataset...")
            fires_df = pd.read_csv(fires_path)
            
            if fires_df.empty:
                logger.error("Fire dataset is empty. Cannot perform spatial statistics.")
                sys.exit(1)
                
            fires_gdf = gpd.GeoDataFrame(
                fires_df, 
                geometry=gpd.points_from_xy(fires_df.longitude, fires_df.latitude),
                crs="EPSG:4326"
            )
            
            engine = SpatialStatisticsEngine()
            final_enriched_gdf, stats_summary = engine.run_master_pipeline(fires_gdf)
            
            out_fires = settings.PROCESSED_DATA_DIR / "master_enriched_fires.csv"
            out_stats = settings.PROCESSED_DATA_DIR / "spatial_stats_summary.json"
            
            # Save enriched dataset
            df_to_save = pd.DataFrame(final_enriched_gdf.drop(columns=['geometry']))
            if 'datetime' in df_to_save.columns:
                df_to_save['datetime'] = df_to_save['datetime'].astype(str)
            df_to_save.to_csv(out_fires, index=False)
            
            # Save stats summary
            with open(out_stats, 'w') as f:
                json.dump(stats_summary, f, indent=4)
                
            logger.info(f"Saved master enriched fires to {out_fires}")
            logger.info(f"Saved spatial stats summary to {out_stats}")
            logger.info("Part 2.4 execution completed successfully.")
        elif args.part == '3':
            from src.training.pipeline import TrainingDataPipeline
            
            logger.info("Starting Part 3: Training Data Preparation Pipeline")
            
            pipeline = TrainingDataPipeline(
                proximity_threshold_km=1.0,
                review_sample_size=50,
                correlation_threshold=0.95,
                smote_strategy="auto",
            )
            result_df = pipeline.run_pipeline()
            
            if result_df.empty:
                logger.error("Training data preparation failed.")
                sys.exit(1)
            
            logger.info("Part 3 execution completed successfully.")
        elif args.part == 'web':
            from src.web.app import create_app
            logger.info(f"Starting Web Dashboard on http://localhost:{args.port}")
            print(f"\n    [+] Dashboard:      http://localhost:{args.port}")
            print(f"    [+] API Stats:      http://localhost:{args.port}/api/stats")
            print(f"    [+] API Fires:      http://localhost:{args.port}/api/fires")
            print(f"    [+] API Facilities: http://localhost:{args.port}/api/facilities\n")
            app = create_app()
            app.run(debug=args.debug, port=args.port, host='0.0.0.0')
        else:
            logger.warning(f"Part {args.part} is not yet implemented.")
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
