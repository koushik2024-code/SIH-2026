"""
Unit and Integration Tests for Part 4.3: Dashboard Controls.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.visualization.interactive_map import InteractiveGISMap
from src.visualization.dashboard_controls import DashboardControlManager
from src.visualization.data_overlays import OverlayManager
from src.web.app import create_app, filter_fire_dataframe
from src.web.demo_data import DemoDataGenerator


class TestPart43DashboardControls(unittest.TestCase):
    """Test suite for Part 4.3 GIS Dashboard Controls and Web Dashboard integration."""

    @classmethod
    def setUpClass(cls):
        demo = DemoDataGenerator()
        cls.facilities_gdf = demo.generate_facilities()
        cls.fire_df = demo.generate_fire_data(n_fires=60, days_back=3)

    def test_dashboard_control_manager_initialization(self):
        """Verify DashboardControlManager properties and constants."""
        mgr = DashboardControlManager()
        self.assertIn("Industrial Fire", mgr.FIRE_TYPE_COLORS)
        self.assertIn("Gas Flare", mgr.FIRE_TYPE_COLORS)
        self.assertEqual(len(mgr.HAZARD_BUFFER_RINGS), 4)

    def test_map_creation_with_controls(self):
        """Verify InteractiveGISMap creates map with Geocoder, filter dock, and legend."""
        gis = InteractiveGISMap(enable_controls=True, enable_temporal_slider=False)
        m = gis.create_interactive_map(
            fire_df=self.fire_df,
            facilities_gdf=self.facilities_gdf,
            enable_controls=True,
        )
        html = m._repr_html_()

        # Check key Part 4.3 controls in rendered HTML
        self.assertTrue(
            "leaflet-control-geocoder" in html or "Geocoder" in html,
            "Geocoder search control missing",
        )
        self.assertIn("gisFilterDock", html, "GIS filter dock missing")
        self.assertIn("dockDateStart", html, "Start date picker missing")
        self.assertIn("dockDateEnd", html, "End date picker missing")
        self.assertIn("dockConfSlider", html, "Confidence slider missing")
        self.assertIn("facSearchInput", html, "Facility search input missing")
        self.assertIn("gisLegendHUD", html, "Legend HUD missing")
        self.assertIn("_fireMarkerRegistry", html, "Fire marker registry script missing")

    def test_filter_fire_dataframe_by_type(self):
        """Test fire type filtering helper."""
        filtered = filter_fire_dataframe(
            self.fire_df,
            selected_types=["Industrial Fire"],
        )
        self.assertFalse(filtered.empty)
        self.assertTrue((filtered["fire_type"] == "Industrial Fire").all())

    def test_filter_fire_dataframe_by_date(self):
        """Test date range filtering."""
        dates = self.fire_df["acq_date"].sort_values()
        min_date = str(dates.iloc[0])
        filtered = filter_fire_dataframe(
            self.fire_df,
            start_date=min_date,
            end_date=min_date,
        )
        self.assertFalse(filtered.empty)
        self.assertTrue((filtered["acq_date"].astype(str) == min_date).all())

    def test_filter_fire_dataframe_by_confidence(self):
        """Test confidence threshold filtering."""
        filtered = filter_fire_dataframe(
            self.fire_df,
            min_confidence=75,
        )
        # Should only contain fires with high / >=75 confidence
        self.assertTrue(len(filtered) <= len(self.fire_df))

    def test_filter_fire_dataframe_by_search(self):
        """Test search query matching."""
        filtered = filter_fire_dataframe(
            self.fire_df,
            search="Industrial",
        )
        self.assertFalse(filtered.empty)

    def test_flask_dashboard_routes_with_filters(self):
        """Verify Flask web dashboard handles Part 4.3 query filters."""
        app = create_app()
        client = app.test_client()

        # 1. Main index with filter query params
        res = client.get("/?fire_type=Industrial+Fire&min_confidence=60")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("Part 4.3 Active", html)
        self.assertIn("sidebarSearch", html)
        self.assertIn("sidebarDateStart", html)
        self.assertIn("sidebarConfSlider", html)

        # 2. Standalone /map endpoint with filter query params
        res_map = client.get("/map?fire_type=Industrial+Fire&min_confidence=60")
        self.assertEqual(res_map.status_code, 200)

        # 3. /api/stats with filter query params
        res_stats = client.get("/api/stats?fire_type=Industrial+Fire")
        self.assertEqual(res_stats.status_code, 200)
        data = res_stats.get_json()
        self.assertIn("total_fires", data)

        # 4. /api/fires with filter query params
        res_fires = client.get("/api/fires?fire_type=Industrial+Fire")
        self.assertEqual(res_fires.status_code, 200)
        fires_list = res_fires.get_json()
        self.assertIsInstance(fires_list, list)


if __name__ == "__main__":
    unittest.main()
