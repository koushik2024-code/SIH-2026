import time
import signal
import sys
import logging
import argparse
from datetime import datetime
from typing import Optional

from src.pipeline_automation.automated_pipeline import AutomatedPipeline

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """
    Recurring task scheduler for the automated thermal anomaly pipeline.
    Default interval is 6 hours (matching NASA FIRMS orbit refresh & NTRO challenge requirements).
    """

    def __init__(self, interval_hours: float = 6.0, simulate: bool = False):
        self.interval_hours = interval_hours
        self.interval_seconds = int(interval_hours * 3600)
        self.simulate = simulate
        self.pipeline = AutomatedPipeline()
        self._running = False

    def start(self, run_immediately: bool = True):
        """
        Start the scheduling loop. Blocks until interrupted.
        """
        self._running = True
        logger.info("===============================================================")
        logger.info(f"   PIPELINE SCHEDULER ACTIVE (Interval: {self.interval_hours} hours)")
        logger.info("   Press Ctrl+C to stop.")
        logger.info("===============================================================")

        # Graceful signal handler
        def _handle_exit(sig, frame):
            logger.info("\nScheduler shutdown signal received. Terminating safely...")
            self._running = False
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, _handle_exit)
            signal.signal(signal.SIGTERM, _handle_exit)
        except Exception:
            pass

        # Immediate first execution
        if run_immediately:
            logger.info("Triggering initial pipeline execution cycle...")
            self.pipeline.run_pipeline(simulate=self.simulate)

        # Check if python 'schedule' library is installed
        use_schedule_lib = False
        try:
            import schedule
            use_schedule_lib = True
        except ImportError:
            use_schedule_lib = False

        if use_schedule_lib:
            self._run_with_schedule_lib()
        else:
            self._run_with_native_loop()

    def _run_with_schedule_lib(self):
        """Scheduler loop using the 'schedule' package."""
        import schedule
        logger.info(f"Configuring 'schedule' library job every {self.interval_hours} hours.")
        
        schedule.every(self.interval_hours).hours.do(
            lambda: self.pipeline.run_pipeline(simulate=self.simulate)
        )

        while self._running:
            schedule.run_pending()
            time.sleep(1)

    def _run_with_native_loop(self):
        """Resilient native loop fallback using time.sleep without external dependencies."""
        logger.info(f"Using native scheduling loop. Next cycle in {self.interval_hours} hours.")
        
        while self._running:
            # Sleep in small slices to remain responsive to termination signals
            wake_time = time.time() + self.interval_seconds
            next_run_str = datetime.fromtimestamp(wake_time).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Next automated cycle scheduled for: {next_run_str}")

            while self._running and time.time() < wake_time:
                time.sleep(1)

            if self._running:
                logger.info(f"Waking up for scheduled execution at {datetime.utcnow().isoformat()}...")
                self.pipeline.run_pipeline(simulate=self.simulate)

def main():
    parser = argparse.ArgumentParser(description="Automated Data Pipeline Scheduler (Part 5.1)")
    parser.add_argument("--interval", type=float, default=6.0, help="Interval in hours between runs (default: 6.0)")
    parser.add_argument("--simulate", action="store_true", help="Force synthetic incremental feed for testing/demo")
    parser.add_argument("--no-immediate", action="store_true", help="Do not run immediately upon launch")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    scheduler = PipelineScheduler(interval_hours=args.interval, simulate=args.simulate)
    scheduler.start(run_immediately=not args.no_immediate)

if __name__ == "__main__":
    main()
