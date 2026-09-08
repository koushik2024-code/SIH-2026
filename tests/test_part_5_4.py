import unittest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.reporting.analyzer import HistoricalAnalyzer
from src.reporting.csv_exporter import CSVReportExporter
from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.html_generator import HTMLReportGenerator
from src.reporting.report_engine import ReportEngine
from src.api.app import app as fastapi_app
from src.web.app import create_app as create_flask_app
from src.web.demo_data import DemoDataGenerator


class TestPart54HistoricalReporting(unittest.TestCase):
    """
    Comprehensive test suite for Part 5.4: Historical Analysis & Reporting.
    Tests monthly/quarterly aggregations, regional trends, facility risk scoring,
    PDF, CSV, HTML export engines, FastAPI endpoints, and Web Dashboard UI.
    """

    @classmethod
    def setUpClass(cls):
        cls.demo_gen = DemoDataGenerator()
        cls.fires_df = cls.demo_gen.generate_fire_data(n_fires=250, days_back=90)
        cls.facilities_gdf = cls.demo_gen.generate_facilities()
        cls.fastapi_client = TestClient(fastapi_app)
        cls.flask_app = create_flask_app()
        cls.flask_client = cls.flask_app.test_client()

    def test_monthly_aggregation(self):
        """Test HistoricalAnalyzer monthly report calculation."""
        analyzer = HistoricalAnalyzer()
        monthly = analyzer.compute_monthly_report(self.fires_df)
        self.assertIsInstance(monthly, list)
        self.assertGreater(len(monthly), 0)

        first_month = monthly[0]
        self.assertIn("period", first_month)
        self.assertIn("total_fires", first_month)
        self.assertIn("avg_brightness_k", first_month)
        self.assertIn("avg_frp_mw", first_month)
        self.assertIn("total_frp_energy_mwh", first_month)
        self.assertIn("mom_growth_percent", first_month)
        self.assertGreater(first_month["total_fires"], 0)

    def test_quarterly_aggregation(self):
        """Test HistoricalAnalyzer quarterly report calculation."""
        analyzer = HistoricalAnalyzer()
        quarterly = analyzer.compute_quarterly_report(self.fires_df)
        self.assertIsInstance(quarterly, list)
        self.assertGreater(len(quarterly), 0)

        first_q = quarterly[0]
        self.assertIn("quarter", first_q)
        self.assertIn("total_fires", first_q)
        self.assertIn("qoq_growth_percent", first_q)
        self.assertIn("fire_type_distribution", first_q)

    def test_regional_trends(self):
        """Test HistoricalAnalyzer regional surveillance categorization."""
        analyzer = HistoricalAnalyzer()
        trends = analyzer.compute_regional_trends(self.fires_df)
        self.assertEqual(len(trends), 4)

        region_names = [r["region_name"] for r in trends]
        self.assertTrue(any("Northern" in n for n in region_names))
        self.assertTrue(any("Western" in n for n in region_names))
        self.assertTrue(any("Central-Eastern" in n for n in region_names))
        self.assertTrue(any("Southern" in n for n in region_names))

        for r in trends:
            self.assertIn("total_fires", r)
            self.assertIn("trend_status", r)

    def test_facility_risk_scores(self):
        """Test HistoricalAnalyzer longitudinal facility risk scoring and tiering."""
        analyzer = HistoricalAnalyzer()
        risks = analyzer.compute_facility_risk_scores(self.fires_df, self.facilities_gdf)
        self.assertIsInstance(risks, list)
        self.assertGreater(len(risks), 0)

        # Facilities must be sorted descending by risk score
        scores = [f["risk_score"] for f in risks]
        self.assertEqual(scores, sorted(scores, reverse=True))

        for f in risks:
            self.assertIn("facility_name", f)
            self.assertIn("risk_score", f)
            self.assertGreaterEqual(f["risk_score"], 0.0)
            self.assertLessEqual(f["risk_score"], 100.0)
            self.assertIn(f["risk_tier"], ["EXTREME RISK", "HIGH RISK", "MODERATE RISK", "LOW RISK"])
            self.assertIn("risk_trajectory", f)

    def test_executive_summary(self):
        """Test HistoricalAnalyzer executive summary generation."""
        analyzer = HistoricalAnalyzer()
        summary = analyzer.generate_executive_summary(self.fires_df, self.facilities_gdf)
        self.assertIn("total_fires_analyzed", summary)
        self.assertIn("total_thermal_energy_mwh", summary)
        self.assertIn("monitored_facilities_count", summary)
        self.assertIn("extreme_risk_facilities", summary)
        self.assertIn("high_risk_facilities", summary)
        self.assertIn("dominant_fire_type", summary)

    def test_csv_export(self):
        """Test CSVReportExporter exports all 4 structured analytical tables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            exporter = CSVReportExporter(output_dir=Path(tmp_dir))
            analyzer = HistoricalAnalyzer()
            monthly = analyzer.compute_monthly_report(self.fires_df)
            quarterly = analyzer.compute_quarterly_report(self.fires_df)
            risks = analyzer.compute_facility_risk_scores(self.fires_df, self.facilities_gdf)
            trends = analyzer.compute_regional_trends(self.fires_df)

            files = exporter.export_all(monthly, quarterly, risks, trends)
            for key, path in files.items():
                self.assertTrue(path.exists(), f"File {path} was not created.")
                self.assertGreater(path.stat().st_size, 0, f"File {path} is empty.")

                # Validate CSV content
                df = pd.read_csv(path)
                self.assertFalse(df.empty)

    def test_pdf_report_generation(self):
        """Test PDFReportGenerator generates a valid non-empty PDF file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_gen = PDFReportGenerator(output_dir=Path(tmp_dir))
            analyzer = HistoricalAnalyzer()
            summary = analyzer.generate_executive_summary(self.fires_df, self.facilities_gdf)
            monthly = analyzer.compute_monthly_report(self.fires_df)
            quarterly = analyzer.compute_quarterly_report(self.fires_df)
            risks = analyzer.compute_facility_risk_scores(self.fires_df, self.facilities_gdf)
            trends = analyzer.compute_regional_trends(self.fires_df)

            pdf_path = pdf_gen.generate_pdf(summary, monthly, quarterly, risks, trends)
            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 500)

            # Check PDF signature
            with open(pdf_path, "rb") as f:
                header = f.read(5)
                self.assertTrue(header.startswith(b"%PDF-") or b"PDF" in header)

    def test_html_report_generation(self):
        """Test HTMLReportGenerator builds standalone executive HTML report."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_gen = HTMLReportGenerator(output_dir=Path(tmp_dir))
            analyzer = HistoricalAnalyzer()
            summary = analyzer.generate_executive_summary(self.fires_df, self.facilities_gdf)
            monthly = analyzer.compute_monthly_report(self.fires_df)
            quarterly = analyzer.compute_quarterly_report(self.fires_df)
            risks = analyzer.compute_facility_risk_scores(self.fires_df, self.facilities_gdf)
            trends = analyzer.compute_regional_trends(self.fires_df)

            html_path = html_gen.generate_html(summary, monthly, quarterly, risks, trends)
            self.assertTrue(html_path.exists())
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("National Thermal Anomaly & Industrial Risk Intelligence Report", content)
                self.assertIn("Facility Longitudinal Risk Matrix", content)
                self.assertIn("@media print", content)

    def test_report_engine_full_run(self):
        """Test ReportEngine coordinates full analytical and export pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = ReportEngine(output_dir=Path(tmp_dir))
            res = engine.run_full_analysis(self.fires_df, self.facilities_gdf)
            self.assertIn("summary", res)
            self.assertIn("files", res)
            self.assertIn("pdf", res["files"])
            self.assertIn("html", res["files"])
            self.assertIn("csv_monthly", res["files"])

    def test_fastapi_report_endpoints(self):
        """Test FastAPI REST endpoints for historical reporting."""
        # 1. Monthly
        res = self.fastapi_client.get("/api/reports/monthly")
        self.assertEqual(res.status_code, 200)
        self.assertIn("data", res.json())

        # 2. Quarterly
        res = self.fastapi_client.get("/api/reports/quarterly")
        self.assertEqual(res.status_code, 200)
        self.assertIn("data", res.json())

        # 3. Regional Trends
        res = self.fastapi_client.get("/api/reports/trends")
        self.assertEqual(res.status_code, 200)
        self.assertIn("data", res.json())

        # 4. Facility Risk
        res = self.fastapi_client.get("/api/reports/facilities/risk")
        self.assertEqual(res.status_code, 200)
        self.assertIn("data", res.json())

        # 5. Summary
        res = self.fastapi_client.get("/api/reports/summary")
        self.assertEqual(res.status_code, 200)
        self.assertIn("summary", res.json())

        # 6. PDF Export
        res = self.fastapi_client.get("/api/reports/export/pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(res.content), 500)

        # 7. HTML Export
        res = self.fastapi_client.get("/api/reports/export/html")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))

    def test_web_reports_ui(self):
        """Test Flask web dashboard /reports console endpoint."""
        res = self.flask_client.get("/reports")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("Historical Fire Surveillance & Facility Risk Dossier", html)
        self.assertIn("Monitored Industrial Facilities Risk Leaderboard", html)
        self.assertIn("Download PDF Report", html)


if __name__ == "__main__":
    unittest.main()
