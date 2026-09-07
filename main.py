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
    parser.add_argument("--part", type=str, choices=['1', '2', '3', '4', '5', 'web'], default='web',
                        help="Which part to run (1-5) or 'web' for dashboard")
    parser.add_argument("--bbox", type=str, help="Bounding box as W,S,E,N", default=None)
    parser.add_argument("--days", type=int, help="Number of days of data to fetch", default=2)
    parser.add_argument("--port", type=int, help="Web server port", default=5000)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--run-once", action='store_true', help="Run single pipeline cycle and exit (Part 5)")
    parser.add_argument("--interval-hours", type=float, default=6.0, help="Interval in hours for scheduler (default: 6.0)")
    parser.add_argument("--simulate", action='store_true', help="Simulate incremental satellite anomalies for demo/testing")
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        if args.part == '1':
            from src.pipeline import DataPipeline
            logger.info("Starting Part 1: Data Ingestion & Preprocessing Pipeline")
            pipeline = DataPipeline()
            pipeline.run_full_pipeline()
            logger.info("Part 1 execution completed successfully.")
        elif args.part == '5':
            from src.pipeline_automation import AutomatedPipeline, PipelineScheduler
            logger.info("Starting Part 5: Monitoring, Alerts & Deployment (5.1 Automated Data Pipeline)")
            if args.run_once:
                logger.info("Running single incremental pipeline pass (--run-once)...")
                pipeline = AutomatedPipeline()
                result = pipeline.run_pipeline(simulate=args.simulate, day_range=args.days)
                if result.get("status") == "SUCCESS":
                    logger.info("Part 5.1 pipeline execution completed successfully.")
                else:
                    logger.error(f"Part 5.1 pipeline execution failed: {result.get('error')}")
            else:
                logger.info(f"Launching recurring task scheduler (interval: {args.interval_hours} hours)...")
                scheduler = PipelineScheduler(interval_hours=args.interval_hours, simulate=args.simulate)
                scheduler.start(run_immediately=True)
        elif args.part == 'web':
            from src.web.app import create_app
            logger.info(f"Starting Web Dashboard on http://localhost:{args.port}")
            print(f"\n    [+] Dashboard:      http://localhost:{args.port}")
            print(f"    [+] API Stats:      http://localhost:{args.port}/api/stats")
            print(f"    [+] API Fires:      http://localhost:{args.port}/api/fires")
            print(f"    [+] API Facilities: http://localhost:{args.port}/api/facilities")
            print(f"    [+] Pipeline Status: http://localhost:{args.port}/api/pipeline/status\n")
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
