import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from config import settings
from src.reporting.analyzer import HistoricalAnalyzer
from src.reporting.csv_exporter import CSVReportExporter
from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.html_generator import HTMLReportGenerator

logger = logging.getLogger(__name__)


class ReportEngine:
    """
    Unified coordinator for Part 5.4: Historical Analysis & Multi-Format Reporting.
    Executes analytical pipelines and builds PDF, CSV, and HTML reports.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or (settings.OUTPUT_DIR / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = HistoricalAnalyzer()
        self.csv_exporter = CSVReportExporter(output_dir=self.output_dir)
        self.pdf_generator = PDFReportGenerator(output_dir=self.output_dir)
        self.html_generator = HTMLReportGenerator(output_dir=self.output_dir)

    def run_full_analysis(self, df: pd.DataFrame, facilities: Any) -> Dict[str, Any]:
        """
        Execute full historical analysis and compile reports in all formats.
        """
        logger.info(f"Running historical analysis on {len(df)} fire records...")

        # 1. Compute analytical datasets
        monthly_data = self.analyzer.compute_monthly_report(df)
        quarterly_data = self.analyzer.compute_quarterly_report(df)
        facility_risks = self.analyzer.compute_facility_risk_scores(df, facilities)
        regional_trends = self.analyzer.compute_regional_trends(df)
        summary = self.analyzer.generate_executive_summary(df, facilities)

        # 2. Export CSVs
        csv_files = self.csv_exporter.export_all(
            monthly_data=monthly_data,
            quarterly_data=quarterly_data,
            facility_risks=facility_risks,
            regional_data=regional_trends
        )

        # 3. Generate PDF Report
        pdf_file = self.pdf_generator.generate_pdf(
            summary=summary,
            monthly_data=monthly_data,
            quarterly_data=quarterly_data,
            facility_risks=facility_risks,
            regional_data=regional_trends
        )

        # 4. Generate HTML Report
        html_file = self.html_generator.generate_html(
            summary=summary,
            monthly_data=monthly_data,
            quarterly_data=quarterly_data,
            facility_risks=facility_risks,
            regional_data=regional_trends
        )

        logger.info("Part 5.4 full historical analysis and reporting completed.")

        return {
            "summary": summary,
            "monthly_data": monthly_data,
            "quarterly_data": quarterly_data,
            "facility_risks": facility_risks,
            "regional_trends": regional_trends,
            "files": {
                "pdf": str(pdf_file),
                "html": str(html_file),
                "csv_monthly": str(csv_files["monthly"]),
                "csv_quarterly": str(csv_files["quarterly"]),
                "csv_facility_risk": str(csv_files["facility_risk"]),
                "csv_regional_trends": str(csv_files["regional_trends"]),
            }
        }
