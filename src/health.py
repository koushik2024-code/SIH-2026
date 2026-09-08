"""
System Health Monitoring & Diagnostics Engine
Part 5.5: Deployment, Containerization & Health Monitoring

Implements liveness probes, readiness probes, and deep system diagnostics
for database integrity, storage health, ML model state, and satellite recency.
"""

import os
import sys
import time
import shutil
import sqlite3
import platform
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from src.core.config import get_config

logger = logging.getLogger(__name__)

# Track process startup timestamp
PROCESS_START_TIME = time.time()


class SystemHealthManager:
    """Manages liveness, readiness, and deep diagnostic health probes."""

    def __init__(self, db_path: Path = None, model_path: Path = None):
        self.config = get_config()
        self.db_path = db_path or self.config.DATABASE_PATH
        self.model_path = model_path or self.config.MODEL_PATH

    @property
    def uptime_seconds(self) -> float:
        """Calculate system process uptime in seconds."""
        return round(time.time() - PROCESS_START_TIME, 2)

    def get_liveness(self) -> Dict[str, Any]:
        """
        Fast liveness probe to verify process execution.

        Returns:
            Dict containing status 'alive', process ID, and uptime.
        """
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "pid": os.getpid(),
        }

    def get_readiness(self) -> Dict[str, Any]:
        """
        Readiness probe verifying database connectivity and basic storage access.

        Returns:
            Dict containing status ('ready' or 'not_ready') and component booleans.
        """
        db_ok = False
        db_error = None

        if self.db_path.exists():
            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=3.0)
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                cursor.fetchone()
                conn.close()
                db_ok = True
            except Exception as e:
                db_error = str(e)
        else:
            # If DB doesn't exist yet, it's considered ready to initialize
            db_ok = True

        data_dir_ok = self.config.DATA_DIR.exists()
        is_ready = db_ok and data_dir_ok

        return {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": {
                "database_accessible": db_ok,
                "data_directory_exists": data_dir_ok,
            },
            "error": db_error,
        }

    def get_deep_diagnostics(self) -> Dict[str, Any]:
        """
        Deep diagnostic inspection analyzing database integrity, table counts,
        disk space, ML model state, telemetry recency, and alert channels.

        Returns:
            Comprehensive structured diagnostics dictionary.
        """
        now = datetime.now(timezone.utc)
        issues = []

        # 1. Database Diagnostics
        db_info = {
            "path": str(self.db_path),
            "exists": self.db_path.exists(),
            "size_kb": round(self.db_path.stat().st_size / 1024, 2) if self.db_path.exists() else 0,
            "connected": False,
            "integrity": "unknown",
            "total_fires": 0,
            "total_facilities": 0,
            "total_alerts": 0,
        }

        if self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("PRAGMA quick_check;")
                check_result = cursor.fetchone()
                db_info["integrity"] = check_result[0] if check_result else "unknown"

                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fire_detections';")
                if cursor.fetchone()[0] > 0:
                    cursor.execute("SELECT COUNT(*) FROM fire_detections;")
                    db_info["total_fires"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='facilities';")
                if cursor.fetchone()[0] > 0:
                    cursor.execute("SELECT COUNT(*) FROM facilities;")
                    db_info["total_facilities"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='alerts';")
                if cursor.fetchone()[0] > 0:
                    cursor.execute("SELECT COUNT(*) FROM alerts;")
                    db_info["total_alerts"] = cursor.fetchone()[0]

                conn.close()
                db_info["connected"] = True
            except Exception as e:
                db_info["error"] = str(e)
                issues.append(f"Database error: {e}")
        else:
            db_info["integrity"] = "missing_file"

        # 2. Storage & Disk Health
        storage_info = {}
        try:
            total, used, free = shutil.disk_usage(self.config.BASE_DIR)
            storage_info = {
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "percent_used": round((used / total) * 100, 1),
                "data_dir_writable": os.access(self.config.DATA_DIR, os.W_OK),
                "output_dir_writable": os.access(self.config.OUTPUT_DIR, os.W_OK),
            }
            if storage_info["percent_used"] > 95:
                issues.append("Low disk space: >95% storage utilized")
        except Exception as e:
            storage_info["error"] = str(e)

        # 3. Machine Learning Model Health
        model_info = {
            "path": str(self.model_path),
            "exists": self.model_path.exists(),
            "size_kb": round(self.model_path.stat().st_size / 1024, 2) if self.model_path.exists() else 0,
            "status": "ready" if self.model_path.exists() else "heuristic_fallback",
        }

        # 4. Telemetry Recency Check
        telemetry_file = self.config.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
        telemetry_info = {
            "file_present": telemetry_file.exists(),
            "path": str(telemetry_file),
            "record_count": 0,
            "last_modified": None,
        }
        if telemetry_file.exists():
            try:
                mtime = datetime.fromtimestamp(telemetry_file.stat().st_mtime, timezone.utc)
                telemetry_info["last_modified"] = mtime.isoformat()
                with open(telemetry_file, "r", encoding="utf-8") as f:
                    # Subtract 1 for header
                    telemetry_info["record_count"] = max(0, sum(1 for _ in f) - 1)
            except Exception as e:
                telemetry_info["error"] = str(e)

        # 5. Alerting Channels Status
        channels = ["console", "websocket_stream"]
        if self.config.ALERT_EMAIL_ENABLED:
            channels.append("email")
        if self.config.ALERT_SMS_ENABLED:
            channels.append("sms")
        if self.config.ALERT_WEBHOOK_URL:
            channels.append("webhook")

        alert_info = {
            "simulation_mode": self.config.ALERT_SIMULATION_MODE,
            "cooldown_minutes": self.config.ALERT_COOLDOWN_MINUTES,
            "active_channels": channels,
        }

        # 6. Overall Health Status
        if len(issues) == 0:
            status = "healthy"
        elif any("Database error" in iss for iss in issues):
            status = "unhealthy"
        else:
            status = "degraded"

        return {
            "status": status,
            "timestamp": now.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "environment": self.config.ENV,
            "system": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
            },
            "database": db_info,
            "storage": storage_info,
            "ml_model": model_info,
            "telemetry": telemetry_info,
            "alerts": alert_info,
            "issues": issues,
        }
