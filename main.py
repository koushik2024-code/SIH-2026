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

    parser = argparse.ArgumentParser(
        description="Industrial Fire Detection System CLI"
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["1", "2", "2.2", "2.3", "2.4", "3", "3.5", "4", "4.1", "4.2", "4.3", "4.4", "5", "5.1", "5.2", "5.3", "5.4", "5.5", "api", "reporting", "deployment", "web"],
        default="web",
        help="Which part to run (1-5, 5.1-5.5, 'api', 'reporting', 'deployment') or 'web' for dashboard",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Trigger immediate diagnostic test alert across all channels (Part 5.2)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as W,S,E,N",
        default=None,
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days of data to fetch",
        default=2,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Web server port",
        default=5000,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run single pipeline cycle and exit (Part 5)",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=6.0,
        help="Interval in hours for scheduler (default: 6.0)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate incremental satellite anomalies for demo/testing",
    )

    args = parser.parse_args()

    print_banner()

    try:
        if args.part == "1":
            from src.pipeline import DataPipeline

            logger.info(
                "Starting Part 1: Data Ingestion & Preprocessing Pipeline"
            )

            pipeline = DataPipeline()
            pipeline.run_full_pipeline()

            logger.info("Part 1 execution completed successfully.")

        elif args.part == "2":
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.analysis.proximity_engine import ProximityAnalysisEngine

            logger.info("Starting Part 2: Proximity Analysis Engine")

            fires_path = (
                settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            )
            facilities_path = (
                settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
            )

            if not fires_path.exists() or not facilities_path.exists():
                logger.error("Missing input data for Part 2. Run Part 1 first.")
                sys.exit(1)

            logger.info("Loading datasets...")

            fires_df = pd.read_csv(fires_path)

            if fires_df.empty:
                logger.error(
                    "Fire dataset is empty. Cannot perform proximity analysis."
                )
                sys.exit(1)

            fires_gdf = gpd.GeoDataFrame(
                fires_df,
                geometry=gpd.points_from_xy(
                    fires_df.longitude,
                    fires_df.latitude,
                ),
                crs="EPSG:4326",
            )

            facilities_gdf = gpd.read_file(facilities_path)

            facility_id_col = (
                "id" if "id" in facilities_gdf.columns else "osm_id"
            )

            if facility_id_col not in facilities_gdf.columns:
                facilities_gdf["facility_id"] = facilities_gdf.index
                facility_id_col = "facility_id"

            if "fire_id" not in fires_gdf.columns:
                fires_gdf["fire_id"] = [
                    f"fire_{i}" for i in range(len(fires_gdf))
                ]

            engine = ProximityAnalysisEngine()

            final_ranked_gdf, buffers_gdf = engine.run_analysis(
                fires_gdf,
                facilities_gdf,
                fire_id_col="fire_id",
                facility_id_col=facility_id_col,
            )

            out_fires = (
                settings.PROCESSED_DATA_DIR / "proximity_scored_fires.csv"
            )
            out_buffers = (
                settings.PROCESSED_DATA_DIR / "facility_buffers.geojson"
            )

            pd.DataFrame(
                final_ranked_gdf.drop(columns=["geometry"])
            ).to_csv(out_fires, index=False)

            buffers_gdf.to_file(out_buffers, driver="GeoJSON")

            logger.info(f"Saved scored fires to {out_fires}")
            logger.info(f"Saved facility buffers to {out_buffers}")
            logger.info("Part 2 execution completed successfully.")

        elif args.part == "2.2":
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.analysis.hotspot_engine import HotspotEngine

            logger.info("Starting Part 2.2: Hotspot Detection Engine")

            fires_path = (
                settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            )

            if not fires_path.exists():
                logger.error(
                    "Missing input data for Part 2.2. Run Part 1 first."
                )
                sys.exit(1)

            logger.info("Loading dataset...")

            fires_df = pd.read_csv(fires_path)

            if fires_df.empty:
                logger.error(
                    "Fire dataset is empty. Cannot perform hotspot analysis."
                )
                sys.exit(1)

            fires_gdf = gpd.GeoDataFrame(
                fires_df,
                geometry=gpd.points_from_xy(
                    fires_df.longitude,
                    fires_df.latitude,
                ),
                crs="EPSG:4326",
            )

            engine = HotspotEngine()
            hotspots_gdf, kde_surface = engine.run_pipeline(fires_gdf)

            out_hotspots = (
                settings.PROCESSED_DATA_DIR / "hotspots_analyzed.csv"
            )
            out_kde = (
                settings.PROCESSED_DATA_DIR / "hotspots_kde_surface.geojson"
            )

            pd.DataFrame(
                hotspots_gdf.drop(columns=["geometry"])
            ).to_csv(out_hotspots, index=False)

            if not kde_surface.empty:
                kde_surface.to_file(out_kde, driver="GeoJSON")

            logger.info(f"Saved analyzed hotspots to {out_hotspots}")
            logger.info(f"Saved KDE heat surface to {out_kde}")
            logger.info("Part 2.2 execution completed successfully.")

        elif args.part == "2.3":
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.analysis.temporal_clustering import (
                TemporalClusteringEngine,
            )

            logger.info("Starting Part 2.3: Temporal Clustering Engine")

            fires_path = (
                settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            )

            if not fires_path.exists():
                logger.error(
                    "Missing input data for Part 2.3. Run Part 1 first."
                )
                sys.exit(1)

            logger.info("Loading dataset...")

            fires_df = pd.read_csv(fires_path)

            if fires_df.empty:
                logger.error(
                    "Fire dataset is empty. Cannot perform temporal clustering."
                )
                sys.exit(1)

            fires_gdf = gpd.GeoDataFrame(
                fires_df,
                geometry=gpd.points_from_xy(
                    fires_df.longitude,
                    fires_df.latitude,
                ),
                crs="EPSG:4326",
            )

            engine = TemporalClusteringEngine()

            clustered_fires_gdf, cluster_hulls_gdf = engine.run_pipeline(
                fires_gdf
            )

            out_fires = (
                settings.PROCESSED_DATA_DIR
                / "temporal_clustered_fires.csv"
            )
            out_hulls = (
                settings.PROCESSED_DATA_DIR
                / "temporal_cluster_hulls.geojson"
            )

            df_to_save = pd.DataFrame(
                clustered_fires_gdf.drop(columns=["geometry"])
            )

            if "datetime" in df_to_save.columns:
                df_to_save["datetime"] = df_to_save["datetime"].astype(str)

            df_to_save.to_csv(out_fires, index=False)

            if not cluster_hulls_gdf.empty:
                hulls_to_save = cluster_hulls_gdf.copy()

                if "first_detection" in hulls_to_save.columns:
                    hulls_to_save["first_detection"] = (
                        hulls_to_save["first_detection"].astype(str)
                    )

                if "last_detection" in hulls_to_save.columns:
                    hulls_to_save["last_detection"] = (
                        hulls_to_save["last_detection"].astype(str)
                    )

                hulls_to_save.to_file(out_hulls, driver="GeoJSON")

            logger.info(f"Saved clustered fires to {out_fires}")
            logger.info(f"Saved cluster hulls to {out_hulls}")
            logger.info("Part 2.3 execution completed successfully.")

        elif args.part == "2.4":
            import json
            import pandas as pd
            import geopandas as gpd

            from config import settings
            from src.analysis.spatial_statistics import (
                SpatialStatisticsEngine,
            )

            logger.info(
                "Starting Part 2.4: Spatial Statistics & Data Enrichment"
            )

            fires_path = (
                settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
            )

            if not fires_path.exists():
                logger.error(
                    "Missing input data for Part 2.4. Run Part 1 first."
                )
                sys.exit(1)

            logger.info("Loading dataset...")

            fires_df = pd.read_csv(fires_path)

            if fires_df.empty:
                logger.error(
                    "Fire dataset is empty. Cannot perform spatial statistics."
                )
                sys.exit(1)

            fires_gdf = gpd.GeoDataFrame(
                fires_df,
                geometry=gpd.points_from_xy(
                    fires_df.longitude,
                    fires_df.latitude,
                ),
                crs="EPSG:4326",
            )

            engine = SpatialStatisticsEngine()

            final_enriched_gdf, stats_summary = (
                engine.run_master_pipeline(fires_gdf)
            )

            out_fires = (
                settings.PROCESSED_DATA_DIR / "master_enriched_fires.csv"
            )
            out_stats = (
                settings.PROCESSED_DATA_DIR / "spatial_stats_summary.json"
            )

            df_to_save = pd.DataFrame(
                final_enriched_gdf.drop(columns=["geometry"])
            )

            if "datetime" in df_to_save.columns:
                df_to_save["datetime"] = df_to_save["datetime"].astype(str)

            df_to_save.to_csv(out_fires, index=False)

            with open(out_stats, "w") as f:
                json.dump(stats_summary, f, indent=4)

            logger.info(f"Saved master enriched fires to {out_fires}")
            logger.info(f"Saved spatial stats summary to {out_stats}")
            logger.info("Part 2.4 execution completed successfully.")

        elif args.part == "3":
            from src.training.pipeline import TrainingDataPipeline

            logger.info(
                "Starting Part 3: Training Data Preparation Pipeline"
            )

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

        elif args.part == "3.5":
            from src.classification.classify_pipeline import (
                ClassificationPipeline,
            )

            logger.info(
                "Starting Part 3.5: Model Training & Evaluation Pipeline"
            )

            pipeline = ClassificationPipeline(
                test_size=0.2,
                cv_folds=5,
                n_iter=50,
                search_method="random",
            )

            results = pipeline.run_training_pipeline()

            if not results:
                logger.error("Model training pipeline failed.")
                sys.exit(1)

            logger.info("Part 3.5 execution completed successfully.")

        elif args.part in ["4", "4.1", "4.2", "4.3"]:
            from src.visualization import InteractiveGISMap
            from config import settings

            is_part_4_3 = (args.part == "4.3")
            is_part_4_2 = (args.part == "4.2")

            if is_part_4_3:
                part_label = "Part 4.3: Dashboard Controls & Interactive GIS Filtering Interface"
                out_filename = "dashboard_map.html"
            elif is_part_4_2:
                part_label = "Part 4.2: Data Overlay Layers & Multi-Temporal GIS Analytics"
                out_filename = "overlay_map.html"
            else:
                part_label = "Part 4.1: GIS Visualization & Interactive Map Engine"
                out_filename = "interactive_map.html"

            logger.info(f"Starting {part_label}")

            fire_df, facilities_gdf = InteractiveGISMap.load_datasets(
                simulate=args.simulate, days=args.days
            )

            if fire_df.empty:
                logger.error("No fire detections available to map.")
                sys.exit(1)

            use_temporal = False if is_part_4_3 else True
            gis_engine = InteractiveGISMap(enable_temporal_slider=use_temporal, enable_controls=True)
            out_output = settings.OUTPUT_DIR / out_filename
            out_interactive = settings.OUTPUT_DIR / "interactive_map.html"
            out_root = settings.BASE_DIR / "map.html"
            out_docs = settings.BASE_DIR / "docs" / "map.html"

            # Export to primary output path
            gis_engine.export_html(
                output_path=out_output,
                fire_df=fire_df,
                facilities_gdf=facilities_gdf,
                enable_temporal_slider=use_temporal,
                enable_controls=True,
            )

            # Ensure interactive_map.html is always kept synchronized
            if out_output != out_interactive:
                gis_engine.export_html(
                    output_path=out_interactive,
                    fire_df=fire_df,
                    facilities_gdf=facilities_gdf,
                    enable_temporal_slider=use_temporal,
                    enable_controls=True,
                )

            # Synchronize to root map.html and docs/map.html for deployment
            gis_engine.export_html(
                output_path=out_root,
                fire_df=fire_df,
                facilities_gdf=facilities_gdf,
                enable_temporal_slider=use_temporal,
                enable_controls=True,
            )

            if out_docs.parent.exists():
                gis_engine.export_html(
                    output_path=out_docs,
                    fire_df=fire_df,
                    facilities_gdf=facilities_gdf,
                    enable_temporal_slider=use_temporal,
                    enable_controls=True,
                )

            part_name = "Part 4.3" if is_part_4_3 else ("Part 4.2" if is_part_4_2 else "Part 4.1")
            logger.info(f"{part_name} Map successfully generated:")
            logger.info(f"  [+] Standalone Output Map:  {out_output}")
            logger.info(f"  [+] Root Application Map:   {out_root}")
            logger.info(f"  [+] GitHub Pages Live Map:  {out_docs}")
            logger.info(f"{part_name} execution completed successfully.")

        elif args.part == "4.4":
            import json
            from src.visualization import InteractiveGISMap, AnalyticsEngine
            from config import settings

            logger.info("Starting Part 4.4: Analytics Panels & Comprehensive Visual Intelligence")

            fire_df, facilities_gdf = InteractiveGISMap.load_datasets(
                simulate=args.simulate, days=args.days
            )

            if fire_df.empty:
                logger.error("No fire detections available for analytics computation.")
                sys.exit(1)

            analytics_engine = AnalyticsEngine()
            
            out_dashboard_html = settings.OUTPUT_DIR / "analytics_dashboard.html"
            out_summary_json = settings.OUTPUT_DIR / "analytics_summary.json"
            out_root_html = settings.BASE_DIR / "analytics.html"
            out_docs_html = settings.BASE_DIR / "docs" / "analytics.html"

            # 1. Generate standalone dashboard HTML
            analytics_engine.generate_standalone_report_html(
                fire_df=fire_df,
                facilities_gdf=facilities_gdf,
                output_path=out_dashboard_html,
            )

            # Synchronize to root analytics.html and docs/analytics.html if directory exists
            analytics_engine.generate_standalone_report_html(
                fire_df=fire_df,
                facilities_gdf=facilities_gdf,
                output_path=out_root_html,
            )
            if out_docs_html.parent.exists():
                analytics_engine.generate_standalone_report_html(
                    fire_df=fire_df,
                    facilities_gdf=facilities_gdf,
                    output_path=out_docs_html,
                )

            # 2. Export structured summary JSON
            analytics_data = analytics_engine.generate_full_analytics(fire_df, facilities_gdf)
            with open(out_summary_json, "w", encoding="utf-8") as f:
                json.dump(analytics_data, f, indent=2)

            # 3. Export static Matplotlib figures
            figures = analytics_engine.export_matplotlib_charts(
                fire_df=fire_df,
                facilities_gdf=facilities_gdf,
                output_dir=settings.OUTPUT_DIR,
            )

            reg = analytics_data["regional_summary"]
            dist = analytics_data["fire_type_distribution"]
            top_facs = analytics_data["top_facilities"]
            ts = analytics_data["time_series"]

            logger.info("=================================================================")
            logger.info("       PART 4.4 ANALYTICS & VISUAL INTELLIGENCE BRIEFING         ")
            logger.info("=================================================================")
            logger.info(f"  [+] Total Fire Detections:          {reg['total_detections']:,}")
            logger.info(f"  [+] Industrial Asset Exposure:      {reg['industrial_fire_count']} ({reg['industrial_exposure_rate']}%)")
            logger.info(f"  [+] Radiative Power (Mean / Peak):  {reg['avg_frp']} MW / {reg['peak_frp']} MW")
            logger.info(f"  [+] High-Confidence Verifications:  {reg['high_confidence_rate']}%")
            logger.info(f"  [+] Time Series Span:               {len(ts['dates'])} observation days")
            logger.info(f"  [+] Monitored Industrial Sites:     {len(top_facs)} priority hazard facilities")
            if top_facs:
                top1 = top_facs[0]
                logger.info(f"  [!] Highest Risk Facility:          {top1['name']} ({top1['fire_count']} fires, min dist: {top1['min_distance_km']} km, risk: {top1.get('risk_level', 'HIGH')})")
            logger.info("-----------------------------------------------------------------")
            logger.info(f"  [+] Standalone Analytics Report:    {out_dashboard_html}")
            logger.info(f"  [+] Structured Analytics JSON:      {out_summary_json}")
            logger.info(f"  [+] Root Deployment Report:         {out_root_html}")
            for fig_key, fig_path in figures.items():
                logger.info(f"  [+] Static Analytics Figure:        {fig_path}")
            logger.info("=================================================================")
            logger.info("Part 4.4 execution completed successfully.")

        elif args.part in ["5", "5.1"]:
            from src.pipeline_automation import (
                AutomatedPipeline,
                PipelineScheduler,
            )

            logger.info(
                "Starting Part 5: Monitoring, Alerts & Deployment "
                "(5.1 Automated Data Pipeline & 5.2 Alert Dispatch)"
            )

            if args.run_once or args.part == "5.1":
                logger.info(
                    "Running single incremental pipeline pass "
                    "(--run-once)..."
                )

                pipeline = AutomatedPipeline()

                result = pipeline.run_pipeline(
                    simulate=args.simulate,
                    day_range=args.days,
                )

                if result.get("status") == "SUCCESS":
                    logger.info(
                        "Part 5.1/5.2 pipeline execution completed successfully. "
                        f"Alerts dispatched: {result.get('alerts_dispatched', 0)}"
                    )
                else:
                    logger.error(
                        "Part 5 pipeline execution failed: "
                        f"{result.get('error')}"
                    )
                    sys.exit(1)

            else:
                logger.info(
                    "Launching recurring task scheduler "
                    f"(interval: {args.interval_hours} hours)..."
                )

                scheduler = PipelineScheduler(
                    interval_hours=args.interval_hours,
                    simulate=args.simulate,
                )

                scheduler.start(run_immediately=True)

        elif args.part == "5.2":
            import pandas as pd
            import geopandas as gpd
            from config import settings
            from src.monitoring import AlertEngine, get_alert_dispatcher
            from src.pipeline_automation.database import FireMonitoringDatabase
            from src.web.demo_data import DemoDataGenerator

            logger.info("===============================================================")
            logger.info("       STARTING PART 5.2: MULTI-CHANNEL ALERT SYSTEM           ")
            logger.info("===============================================================")

            db = FireMonitoringDatabase()
            dispatcher = get_alert_dispatcher()
            dispatcher.set_database(db)
            alert_engine = AlertEngine(dispatcher=dispatcher, db=db)

            # 1. Check if diagnostic test alert requested
            if args.test_alert:
                logger.info("Dispatching diagnostic test alert across all channels...")
                test_alert = alert_engine.trigger_test_alert()
                logger.info(f"Test alert [{test_alert.severity}] '{test_alert.title}' dispatched to: {test_alert.channels_dispatched}")
                logger.info("Part 5.2 diagnostic test completed successfully.")
                return

            # 2. Load fire dataset and facilities for evaluation
            fires_df = db.get_all_fires(limit=1000)
            if fires_df.empty or args.simulate:
                logger.info("Evaluating alert triggers using simulated / baseline satellite detections...")
                demo_gen = DemoDataGenerator()
                fires_df = demo_gen.generate_fire_data(n_fires=250, days_back=3)
                facilities_gdf = demo_gen.generate_facilities()
            else:
                fac_path = settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
                facilities_gdf = gpd.read_file(fac_path) if fac_path.exists() else None

            # 3. Run trigger evaluation and dispatch
            alerts = alert_engine.evaluate_and_dispatch(
                df=fires_df,
                facilities_gdf=facilities_gdf,
                historical_df=fires_df,
                check_cooldown=not args.simulate,
            )

            # 4. Print Executive Operational Briefing
            alert_stats = db.get_alert_statistics()
            logger.info("---------------------------------------------------------------")
            logger.info("           PART 5.2 INCIDENT DISPATCH SUMMARY                 ")
            logger.info("---------------------------------------------------------------")
            logger.info(f"  [+] Active Detections Evaluated:  {len(fires_df)}")
            logger.info(f"  [+] Dispatched Alert Events:      {len(alerts)}")
            logger.info(f"  [+] Persistent Total in DB:       {alert_stats['total_alerts']}")
            logger.info(f"  [+] Critical Active Incidents:    {alert_stats['critical_active_alerts']}")
            logger.info(f"  [+] Active Channels:              Email (SMTP), SMS (Twilio), Dashboard (SSE), Log, Webhook")
            for idx, a in enumerate(alerts[:5], 1):
                logger.info(f"      {idx}. [{a.severity}] {a.title} -> {a.channels_dispatched}")
            if len(alerts) > 5:
                logger.info(f"      ... and {len(alerts) - 5} more alert(s)")
            logger.info("---------------------------------------------------------------")
            logger.info(f"  [+] Audit Log File:  {settings.ALERT_LOG_PATH}")
            logger.info(f"  [+] JSON Records:    {settings.ALERT_JSON_PATH}")
            logger.info("===============================================================")
            logger.info("Part 5.2 execution completed successfully.")

        elif args.part in ["5.3", "api"]:
            import uvicorn

            api_port = args.port if args.port != 5000 else 8000
            logger.info("===============================================================")
            logger.info("        STARTING PART 5.3: PRODUCTION FASTAPI REST API        ")
            logger.info("===============================================================")
            print(f"\n    [+] FastAPI Server:    http://localhost:{api_port}")
            print(f"    [+] Interactive Docs:  http://localhost:{api_port}/docs")
            print(f"    [+] ReDoc:             http://localhost:{api_port}/redoc")
            print(f"    [+] API Fires:         http://localhost:{api_port}/api/fires")
            print(f"    [+] API Facilities:    http://localhost:{api_port}/api/facilities")
            print(f"    [+] API Hotspots:      http://localhost:{api_port}/api/hotspots")
            print(f"    [+] API Stats:         http://localhost:{api_port}/api/stats")
            print(f"    [+] API Classify:      POST http://localhost:{api_port}/api/classify")
            print(f"    [+] API Health:        http://localhost:{api_port}/health\n")

            from src.api.app import app
            uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")

        elif args.part in ["5.4", "reporting"]:
            logger.info("===============================================================")
            logger.info("   STARTING PART 5.4: HISTORICAL ANALYSIS & REPORTING ENGINE   ")
            logger.info("===============================================================")
            from src.web.app import get_data
            fires_df, facilities_gdf = get_data()

            from src.reporting.report_engine import ReportEngine
            engine = ReportEngine()
            analysis = engine.run_full_analysis(fires_df, facilities_gdf)
            summary = analysis["summary"]

            logger.info("---------------------------------------------------------------")
            logger.info("               EXECUTIVE REPORTING DOSSIER                    ")
            logger.info("---------------------------------------------------------------")
            logger.info(f"  [+] Total Fire Detections:       {summary.get('total_fires_analyzed')}")
            logger.info(f"  [+] Total Thermal Energy:        {summary.get('total_thermal_energy_mwh')} MWh")
            logger.info(f"  [+] Monitored Facilities:        {summary.get('monitored_facilities_count')}")
            logger.info(f"  [+] Extreme / High Risk Assets:  {summary.get('extreme_risk_facilities')} Extreme, {summary.get('high_risk_facilities')} High")
            logger.info(f"  [+] Monthly Periods Aggregated:  {summary.get('monthly_periods')}")
            logger.info(f"  [+] Regional Surveillance Zones: {summary.get('regional_zones')}")
            logger.info("---------------------------------------------------------------")
            logger.info("               GENERATED REPORT ARTIFACTS                      ")
            logger.info("---------------------------------------------------------------")
            for fmt, path in analysis.get("files", {}).items():
                logger.info(f"  [+] {fmt.upper()}: {path}")
            logger.info("===============================================================")
            logger.info("Part 5.4 execution completed successfully.")

        elif args.part in ["5.5", "deployment"]:
            logger.info("===============================================================")
            logger.info("   STARTING PART 5.5: DEPLOYMENT PRE-FLIGHT VERIFICATION       ")
            logger.info("===============================================================")
            from scripts.deploy import run_preflight
            success = run_preflight()
            if success:
                logger.info("Part 5.5 pre-flight verification completed successfully.")
            else:
                logger.error("Part 5.5 pre-flight verification encountered critical errors.")
                sys.exit(1)

        elif args.part == "web":
            from src.web.app import create_app

            logger.info(
                f"Starting Web Dashboard on http://localhost:{args.port}"
            )

            print(
                f"\n    [+] Dashboard:      "
                f"http://localhost:{args.port}"
            )
            print(
                f"    [+] API Stats:      "
                f"http://localhost:{args.port}/api/stats"
            )
            print(
                f"    [+] API Fires:      "
                f"http://localhost:{args.port}/api/fires"
            )
            print(
                f"    [+] API Facilities: "
                f"http://localhost:{args.port}/api/facilities"
            )
            print(
                f"    [+] Pipeline Status: "
                f"http://localhost:{args.port}/api/pipeline/status\n"
            )

            app = create_app()

            app.run(
                debug=args.debug,
                port=args.port,
                host="0.0.0.0",
            )

        else:
            logger.warning(
                f"Part {args.part} is not yet implemented."
            )

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
        sys.exit(1)

    except Exception as e:
        logger.exception(
            f"An unexpected error occurred: {e}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
