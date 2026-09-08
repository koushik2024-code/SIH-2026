"""
Unit and Integration Tests for Part 4.4: Analytics Panels.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.visualization.analytics_panels import AnalyticsEngine
from src.web.app import create_app
from src.web.demo_data import DemoDataGenerator


class TestPart44AnalyticsPanels(unittest.TestCase):
    """Test suite for Part 4.4 Analytics Panels and Visual Intelligence."""

    @classmethod
    def setUpClass(cls):
        demo = DemoDataGenerator()
        cls.facilities_gdf = demo.generate_facilities()
        cls.fire_df = demo.generate_fire_data(n_fires=80, days_back=5)
        cls.engine = AnalyticsEngine()
        cls.flask_app = create_app()
        cls.flask_client = cls.flask_app.test_client()
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_engine_initialization(self):
        """Verify AnalyticsEngine initializes with required color mappings."""
        self.assertIn("Industrial Fire", self.engine.FIRE_TYPE_COLORS)
        self.assertIn("Gas Flare", self.engine.FIRE_TYPE_COLORS)
        self.assertIn("Forest Fire", self.engine.FIRE_TYPE_COLORS)
        self.assertIn("Agricultural Burning", self.engine.FIRE_TYPE_COLORS)

    def test_compute_fire_type_distribution(self):
        """Verify Part 4.4.1 fire type distribution computation."""
        dist = self.engine.compute_fire_type_distribution(self.fire_df)
        self.assertIn("labels", dist)
        self.assertIn("counts", dist)
        self.assertIn("percentages", dist)
        self.assertIn("colors", dist)
        self.assertIn("table", dist)
        self.assertEqual(sum(dist["counts"]), len(self.fire_df))
        self.assertAlmostEqual(sum(dist["percentages"]), 100.0, delta=1.0)
        self.assertTrue(len(dist["table"]) > 0)
        first_row = dist["table"][0]
        self.assertIn("category", first_row)
        self.assertIn("count", first_row)
        self.assertIn("percentage", first_row)
        self.assertIn("avg_frp", first_row)

    def test_compute_time_series(self):
        """Verify Part 4.4.2 chronological time series aggregation."""
        ts = self.engine.compute_time_series(self.fire_df)
        self.assertIn("dates", ts)
        self.assertIn("counts", ts)
        self.assertIn("total_frp", ts)
        self.assertIn("avg_frp", ts)
        self.assertIn("cumulative_counts", ts)
        self.assertEqual(len(ts["dates"]), len(ts["counts"]))
        self.assertEqual(len(ts["dates"]), len(ts["total_frp"]))
        self.assertEqual(sum(ts["counts"]), len(self.fire_df))
        if ts["cumulative_counts"]:
            self.assertEqual(ts["cumulative_counts"][-1], len(self.fire_df))

    def test_compute_top_facilities(self):
        """Verify Part 4.4.3 top facilities hazard leaderboard computation."""
        top_facs = self.engine.compute_top_facilities(
            self.fire_df, self.facilities_gdf, max_dist_km=5.0, top_n=10
        )
        self.assertIsInstance(top_facs, list)
        self.assertTrue(len(top_facs) <= 10)
        if top_facs:
            first_fac = top_facs[0]
            self.assertIn("name", first_fac)
            self.assertIn("facility_type", first_fac)
            self.assertIn("fire_count", first_fac)
            self.assertIn("min_distance_km", first_fac)
            self.assertIn("risk_level", first_fac)
            self.assertIn(first_fac["risk_level"], ["CRITICAL", "HIGH", "MEDIUM", "LOW"])

    def test_compute_regional_summary(self):
        """Verify Part 4.4.4 regional surveillance statistics and quadrant risk."""
        reg = self.engine.compute_regional_summary(self.fire_df, self.facilities_gdf)
        self.assertEqual(reg["total_detections"], len(self.fire_df))
        self.assertIn("industrial_fire_count", reg)
        self.assertIn("industrial_exposure_rate", reg)
        self.assertIn("avg_frp", reg)
        self.assertIn("peak_frp", reg)
        self.assertIn("daytime_count", reg)
        self.assertIn("nighttime_count", reg)
        self.assertEqual(reg["daytime_count"] + reg["nighttime_count"], len(self.fire_df))
        self.assertIn("quadrants", reg)
        self.assertGreaterEqual(len(reg["quadrants"]), 1)

    def test_generate_full_analytics(self):
        """Verify master payload generator contains all necessary sections."""
        payload = self.engine.generate_full_analytics(self.fire_df, self.facilities_gdf)
        self.assertIn("distribution", payload)
        self.assertIn("fire_type_distribution", payload)
        self.assertIn("time_series", payload)
        self.assertIn("top_facilities", payload)
        self.assertIn("regional", payload)
        self.assertIn("regional_summary", payload)

    def test_generate_standalone_report_html(self):
        """Verify standalone report HTML export with Chart.js canvas elements."""
        out_html = self.temp_dir / "test_report.html"
        self.engine.generate_standalone_report_html(
            self.fire_df, self.facilities_gdf, out_html
        )
        self.assertTrue(out_html.exists())
        content = out_html.read_text(encoding="utf-8")
        self.assertIn("fireTypeDonut", content)
        self.assertIn("timeSeriesChart", content)
        self.assertIn("Spatial Analytics & Intelligence Report", content)
        self.assertIn("Part 4.4", content)

    def test_export_matplotlib_charts(self):
        """Verify static PNG chart generation."""
        figs = self.engine.export_matplotlib_charts(
            self.fire_df, self.facilities_gdf, self.temp_dir
        )
        self.assertIn("fire_type_distribution", figs)
        self.assertIn("time_series", figs)
        self.assertIn("top_facilities", figs)
        self.assertIn("regional_quadrants", figs)
        for name, p in figs.items():
            self.assertTrue(p.exists(), f"Static figure {name} missing: {p}")
            self.assertGreater(p.stat().st_size, 1000)

    def test_flask_analytics_page(self):
        """Verify Flask web route /analytics renders successfully."""
        resp = self.flask_client.get("/analytics")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Fire Spatial Intelligence & Analytics", html)
        self.assertIn("Part 4.4 Active", html)
        self.assertIn("fireTypeDonut", html)
        self.assertIn("timeSeriesChart", html)
        self.assertIn("facilitiesBarChart", html)
        self.assertIn("diurnalDonut", html)

    def test_flask_api_analytics(self):
        """Verify Flask API route /api/analytics returns valid JSON schema."""
        resp = self.flask_client.get("/api/analytics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("fire_type_distribution", data)
        self.assertIn("time_series", data)
        self.assertIn("top_facilities", data)
        self.assertIn("regional_summary", data)


if __name__ == "__main__":
    unittest.main()
