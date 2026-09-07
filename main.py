import argparse
import logging
import sys

from src.pipeline import DataPipeline

def setup_logging():
    """Setup basic logging for the entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def print_banner():
    """Print project banner."""
    banner = """
    =================================================
       INDUSTRIAL FIRE DETECTION SYSTEM (SIH-2026)
    =================================================
    """
    print(banner)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Industrial Fire Detection System CLI")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help="Which part of the project to run (1-5)")
    parser.add_argument("--bbox", type=str, help="Bounding box as W,S,E,N", default=None)
    parser.add_argument("--days", type=int, help="Number of days of data to fetch", default=2)
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        if args.part == 1:
            logger.info("Starting Part 1: Data Ingestion & Preprocessing Pipeline")
            pipeline = DataPipeline()
            # Note: arguments like bbox and days could be passed to pipeline here in a more advanced implementation
            pipeline.run_full_pipeline()
            logger.info("Part 1 execution completed successfully.")
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
