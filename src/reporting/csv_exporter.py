import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class CSVReportExporter:
    """
    Exports analytical datasets and summary tables into standardized CSV files.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or (settings.OUTPUT_DIR / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_monthly_summary(self, monthly_data: List[Dict[str, Any]], filename: str = "monthly_fire_summary.csv") -> Path:
        """Export monthly aggregated metrics to CSV."""
        path = self.output_dir / filename
        if not monthly_data:
            df = pd.DataFrame(columns=["period", "total_fires", "avg_brightness_k", "avg_frp_mw", "mom_growth_percent"])
        else:
            # Flatten fire_type_distribution dict into string or separate columns
            flattened = []
            for m in monthly_data:
                item = dict(m)
                dist = item.pop("fire_type_distribution", {})
                for k, v in dist.items():
                    item[f"type_{k.lower().replace(' ', '_')}"] = v
                flattened.append(item)
            df = pd.DataFrame(flattened)

        df.to_csv(path, index=False)
        logger.info(f"Exported monthly summary CSV to {path}")
        return path

    def export_quarterly_summary(self, quarterly_data: List[Dict[str, Any]], filename: str = "quarterly_fire_summary.csv") -> Path:
        """Export quarterly aggregated metrics to CSV."""
        path = self.output_dir / filename
        if not quarterly_data:
            df = pd.DataFrame(columns=["quarter", "total_fires", "avg_brightness_k", "avg_frp_mw", "qoq_growth_percent"])
        else:
            flattened = []
            for q in quarterly_data:
                item = dict(q)
                dist = item.pop("fire_type_distribution", {})
                for k, v in dist.items():
                    item[f"type_{k.lower().replace(' ', '_')}"] = v
                flattened.append(item)
            df = pd.DataFrame(flattened)

        df.to_csv(path, index=False)
        logger.info(f"Exported quarterly summary CSV to {path}")
        return path

    def export_facility_risk_scores(self, facility_risks: List[Dict[str, Any]], filename: str = "facility_risk_scores.csv") -> Path:
        """Export facility longitudinal risk ratings to CSV."""
        path = self.output_dir / filename
        df = pd.DataFrame(facility_risks) if facility_risks else pd.DataFrame(
            columns=["facility_id", "facility_name", "facility_type", "risk_score", "risk_tier"]
        )
        df.to_csv(path, index=False)
        logger.info(f"Exported facility risk scores CSV to {path}")
        return path

    def export_regional_trends(self, regional_data: List[Dict[str, Any]], filename: str = "regional_trends.csv") -> Path:
        """Export regional surveillance trend statistics to CSV."""
        path = self.output_dir / filename
        df = pd.DataFrame(regional_data) if regional_data else pd.DataFrame(
            columns=["region_id", "region_name", "total_fires", "trend_status"]
        )
        df.to_csv(path, index=False)
        logger.info(f"Exported regional trends CSV to {path}")
        return path

    def export_all(
        self,
        monthly_data: List[Dict[str, Any]],
        quarterly_data: List[Dict[str, Any]],
        facility_risks: List[Dict[str, Any]],
        regional_data: List[Dict[str, Any]]
    ) -> Dict[str, Path]:
        """Export all 4 analytical tables simultaneously."""
        return {
            "monthly": self.export_monthly_summary(monthly_data),
            "quarterly": self.export_quarterly_summary(quarterly_data),
            "facility_risk": self.export_facility_risk_scores(facility_risks),
            "regional_trends": self.export_regional_trends(regional_data),
        }
