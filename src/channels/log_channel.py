import json
import logging
from pathlib import Path
from typing import Optional

from config import settings
from src.monitoring.models import AlertEvent
from src.monitoring.channels.base import BaseAlertChannel

logger = logging.getLogger(__name__)

class LogAlertChannel(BaseAlertChannel):
    """
    Appends structured JSONL audit entries to data/alerts.log and
    exports active alert records to output/alerts.json.
    """

    def __init__(self, log_path: Optional[Path] = None, json_path: Optional[Path] = None):
        super().__init__(name="log")
        self.log_path = Path(log_path or settings.ALERT_LOG_PATH)
        self.json_path = Path(json_path or settings.ALERT_JSON_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    def is_enabled(self) -> bool:
        return True

    def send(self, alert: AlertEvent) -> bool:
        """Write alert event to disk audit files."""
        alert_dict = alert.to_dict()

        # 1. Append JSONL line to alerts.log
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_dict) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to alert log file {self.log_path}: {e}")

        # 2. Update output/alerts.json list
        try:
            current_alerts = []
            if self.json_path.exists():
                try:
                    with open(self.json_path, "r", encoding="utf-8") as f:
                        current_alerts = json.load(f)
                        if not isinstance(current_alerts, list):
                            current_alerts = []
                except Exception:
                    current_alerts = []

            # Prepend latest alert and keep last 200
            current_alerts.insert(0, alert_dict)
            current_alerts = current_alerts[:200]

            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(current_alerts, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update {self.json_path}: {e}")

        logger.info(f"Audit log updated for alert [{alert.severity}] {alert.alert_id}")
        return True
