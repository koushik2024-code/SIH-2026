"""
Core Configuration & Environment Management Module
Part 5.5: Deployment, Containerization & Health Monitoring

Provides strongly typed configuration from environment variables,
path resolutions, and comprehensive pre-flight environment validation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

logger = logging.getLogger(__name__)

# Base repository root
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class AppConfig:
    """Production application settings and validation manager."""

    def __init__(self):
        # Operational Environment
        self.ENV: str = os.getenv("APP_ENV", "production").lower()
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # Service Networking
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
        self.WEB_HOST: str = os.getenv("WEB_HOST", "0.0.0.0")
        self.WEB_PORT: int = int(os.getenv("WEB_PORT", os.getenv("PORT", "5000")))

        # Core Storage Paths
        self.BASE_DIR: Path = BASE_DIR
        self.DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
        self.PROCESSED_DATA_DIR: Path = self.DATA_DIR / "processed"
        self.RAW_DATA_DIR: Path = self.DATA_DIR / "raw"
        self.OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
        self.REPORTS_DIR: Path = self.OUTPUT_DIR / "reports"
        self.MODELS_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "models")
        self.DATABASE_PATH: Path = BASE_DIR / os.getenv("DATABASE_PATH", "data/fire_monitoring.db")
        self.MODEL_PATH: Path = self.MODELS_DIR / "fire_classifier.joblib"

        # Satellite Telemetry & Pipeline
        self.FIRMS_MAP_KEY: str = os.getenv("FIRMS_MAP_KEY", "")
        self.SATELLITE_FETCH_DAYS: int = int(os.getenv("SATELLITE_FETCH_DAYS", "2"))
        self.PIPELINE_INTERVAL_HOURS: float = float(os.getenv("PIPELINE_INTERVAL_HOURS", "6.0"))

        # Alerting Subsystem
        self.ALERT_SIMULATION_MODE: bool = os.getenv("ALERT_SIMULATION_MODE", "true").lower() in ("true", "1", "yes")
        self.ALERT_COOLDOWN_MINUTES: int = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))
        self.ALERT_EMAIL_ENABLED: bool = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() in ("true", "1", "yes")
        self.ALERT_SMS_ENABLED: bool = os.getenv("ALERT_SMS_ENABLED", "false").lower() in ("true", "1", "yes")
        self.ALERT_WEBHOOK_URL: str = os.getenv("ALERT_WEBHOOK_URL", "")

        # Docker & Security
        self.CORS_ORIGINS: List[str] = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        for directory in [
            self.DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.RAW_DATA_DIR,
            self.OUTPUT_DIR,
            self.REPORTS_DIR,
            self.MODELS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_environment(self) -> Dict[str, Any]:
        """
        Execute comprehensive pre-flight sanity checks on runtime environment.

        Returns:
            Dict containing boolean is_valid, list of errors, list of warnings,
            and dictionary of diagnostic details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # 1. Directory Checks & Write Permission
        dir_status = {}
        for name, pth in [
            ("data", self.DATA_DIR),
            ("output", self.OUTPUT_DIR),
            ("models", self.MODELS_DIR),
            ("reports", self.REPORTS_DIR),
        ]:
            exists = pth.exists()
            writable = False
            if exists:
                test_file = pth / f".write_test_{os.getpid()}"
                try:
                    test_file.write_text("ok", encoding="utf-8")
                    test_file.unlink()
                    writable = True
                except Exception as e:
                    writable = False
            else:
                try:
                    pth.mkdir(parents=True, exist_ok=True)
                    exists = True
                    writable = True
                except Exception as e:
                    exists = False
                    writable = False

            dir_status[name] = {"path": str(pth), "exists": exists, "writable": writable}
            if not exists:
                errors.append(f"Directory {name} does not exist and cannot be created: {pth}")
            elif not writable:
                errors.append(f"Directory {name} is not writable: {pth}")

        details["directories"] = dir_status

        # 2. Database Status
        db_exists = self.DATABASE_PATH.exists()
        db_size_kb = round(self.DATABASE_PATH.stat().st_size / 1024, 2) if db_exists else 0
        details["database"] = {
            "path": str(self.DATABASE_PATH),
            "exists": db_exists,
            "size_kb": db_size_kb,
        }
        if not db_exists:
            warnings.append(
                f"SQLite database not found at {self.DATABASE_PATH}. "
                "Will be auto-initialized during startup or demo data generation."
            )

        # 3. Machine Learning Model Check
        model_exists = self.MODEL_PATH.exists()
        details["model"] = {
            "path": str(self.MODEL_PATH),
            "exists": model_exists,
            "size_kb": round(self.MODEL_PATH.stat().st_size / 1024, 2) if model_exists else 0,
        }
        if not model_exists:
            warnings.append(
                f"Trained ML model not found at {self.MODEL_PATH}. "
                "Inference will run in heuristic/fallback classification mode."
            )

        # 4. Satellite FIRMS API Key Check
        has_firms_key = bool(self.FIRMS_MAP_KEY and self.FIRMS_MAP_KEY != "your_api_key_here")
        details["firms_api"] = {
            "key_configured": has_firms_key,
            "key_preview": f"{self.FIRMS_MAP_KEY[:4]}***" if has_firms_key else "None",
        }
        if not has_firms_key:
            warnings.append(
                "NASA FIRMS API key not set or is placeholder. "
                "System will operate using cached data and simulated telemetry."
            )

        # 5. Port Configuration Sanity
        if not (1024 <= self.API_PORT <= 65535):
            errors.append(f"Invalid API_PORT: {self.API_PORT}. Must be between 1024 and 65535.")
        if not (1024 <= self.WEB_PORT <= 65535):
            errors.append(f"Invalid WEB_PORT: {self.WEB_PORT}. Must be between 1024 and 65535.")
        if self.API_PORT == self.WEB_PORT:
            errors.append(f"Port conflict: API_PORT and WEB_PORT are both set to {self.API_PORT}.")

        details["ports"] = {
            "api_port": self.API_PORT,
            "web_port": self.WEB_PORT,
        }

        # 6. Overall validity
        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "details": details,
            "environment": self.ENV,
            "debug": self.DEBUG,
        }


# Global singleton instance
_config_instance: AppConfig = None


def get_config() -> AppConfig:
    """Retrieve or initialize the global AppConfig singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance
