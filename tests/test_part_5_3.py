import unittest
import json
from fastapi.testclient import TestClient

from src.api.app import app

class TestPart53RestAPI(unittest.TestCase):
    """
    Comprehensive test suite for Part 5.3: Production REST API (FastAPI).
    Tests all required endpoints: fires, facilities, hotspots, stats, classify, and health.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns API metadata and documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("title", data)
        self.assertIn("docs", data)
        self.assertEqual(data["docs"], "/docs")
        self.assertIn("endpoints", data)

    def test_health_endpoints(self):
        """Test GET /health and GET /api/health."""
        for path in ["/health", "/api/health"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn(data["status"], ["ok", "degraded"])
            self.assertIn("database_connected", data)
            self.assertIn("total_fires_in_db", data)
            self.assertIn("ml_classifier_loaded", data)

    def test_get_fires_paginated(self):
        """Test GET /api/fires pagination and response schema."""
        response = self.client.get("/api/fires?limit=5&offset=0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)
        self.assertIn("count", data)
        self.assertEqual(data["limit"], 5)
        self.assertEqual(data["offset"], 0)
        self.assertIsInstance(data["data"], list)
        self.assertLessEqual(len(data["data"]), 5)

        if data["data"]:
            item = data["data"][0]
            self.assertIn("detection_id", item)
            self.assertIn("latitude", item)
            self.assertIn("longitude", item)
            self.assertIn("fire_type", item)

    def test_get_fires_with_filters(self):
        """Test GET /api/fires with fire_type and min_confidence filters."""
        response = self.client.get("/api/fires?fire_type=Industrial+Fire&min_confidence=70")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["data"], list)
        for item in data["data"]:
            self.assertEqual(item["fire_type"].lower(), "industrial fire")

    def test_get_fire_by_id_success_and_404(self):
        """Test GET /api/fires/{detection_id} for existing record and 404 for missing."""
        # 1. Fetch a fire to obtain a valid ID
        list_res = self.client.get("/api/fires?limit=1")
        self.assertEqual(list_res.status_code, 200)
        list_data = list_res.json()

        if list_data["data"]:
            valid_id = list_data["data"][0]["detection_id"]
            res = self.client.get(f"/api/fires/{valid_id}")
            self.assertEqual(res.status_code, 200)
            single_data = res.json()
            self.assertEqual(single_data["detection_id"], valid_id)

        # 2. Test 404
        missing_res = self.client.get("/api/fires/NON_EXISTENT_ID_999999")
        self.assertEqual(missing_res.status_code, 404)
        self.assertIn("not found", missing_res.json()["detail"].lower())

    def test_get_fires_by_type_endpoint(self):
        """Test GET /api/fires/type/{fire_type} endpoint."""
        response = self.client.get("/api/fires/type/Gas%20Flare?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["data"], list)
        for item in data["data"]:
            self.assertEqual(item["fire_type"].lower(), "gas flare")

    def test_get_facilities(self):
        """Test GET /api/facilities returns catalog of industrial facilities."""
        response = self.client.get("/api/facilities?limit=20")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)
        self.assertGreater(data["total"], 0)
        self.assertIsInstance(data["data"], list)

        first_fac = data["data"][0]
        self.assertIn("facility_id", first_fac)
        self.assertIn("name", first_fac)
        self.assertIn("facility_type", first_fac)
        self.assertIn("latitude", first_fac)
        self.assertIn("longitude", first_fac)

    def test_get_facilities_filtered(self):
        """Test GET /api/facilities?facility_type=... filtering."""
        response = self.client.get("/api/facilities?facility_type=oil_refinery")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["data"], list)
        for fac in data["data"]:
            self.assertEqual(fac["facility_type"].lower(), "oil_refinery")

    def test_get_hotspots(self):
        """Test GET /api/hotspots returns ranked high-intensity thermal anomalies."""
        response = self.client.get("/api/hotspots?min_frp=15.0&limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)
        self.assertIsInstance(data["data"], list)

        if len(data["data"]) >= 2:
            # Hotspots must be sorted descending by intensity_score
            scores = [h["intensity_score"] for h in data["data"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

        if data["data"]:
            h = data["data"][0]
            self.assertIn("detection_id", h)
            self.assertIn("intensity_score", h)
            self.assertIn("frp", h)
            self.assertIn("brightness", h)

    def test_get_stats(self):
        """Test GET /api/stats returns comprehensive analytics statistics."""
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_fires", data)
        self.assertGreater(data["total_fires"], 0)
        self.assertIn("avg_brightness", data)
        self.assertIn("avg_frp", data)
        self.assertIn("fire_type_distribution", data)
        self.assertIn("day_fires", data)
        self.assertIn("night_fires", data)
        self.assertIn("near_industrial_count", data)

    def test_post_classify_single_observation(self):
        """Test POST /api/classify with a single fire observation payload."""
        payload = {
            "latitude": 22.4707,
            "longitude": 70.0577,
            "brightness": 385.0,
            "frp": 65.0,
            "confidence": 95.0,
            "distance_to_nearest_industrial": 0.4,
            "nearest_facility_type": "oil_refinery"
        }
        response = self.client.post("/api/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_classified"], 1)
        res = data["results"][0]
        self.assertIn(res["fire_type"], ["Industrial Fire", "Gas Flare", "Forest Fire", "Agricultural Burning", "Mining Activity", "Other/Unknown"])
        self.assertIn("classification_confidence", res)
        self.assertGreater(res["classification_confidence"], 0.0)

    def test_post_classify_batch(self):
        """Test POST /api/classify with batch records."""
        payload = {
            "records": [
                {
                    "latitude": 22.4707,
                    "longitude": 70.0577,
                    "brightness": 395.0,
                    "frp": 80.0,
                    "confidence": 92.0,
                    "distance_to_nearest_industrial": 0.2,
                    "nearest_facility_type": "petrochemical"
                },
                {
                    "latitude": 30.0,
                    "longitude": 79.0,
                    "brightness": 330.0,
                    "frp": 70.0,
                    "confidence": 85.0,
                    "distance_to_nearest_industrial": 25.0,
                    "nearest_facility_type": "none"
                }
            ]
        }
        response = self.client.post("/api/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_classified"], 2)
        self.assertEqual(len(data["results"]), 2)

    def test_openapi_documentation(self):
        """Test OpenAPI schema and Swagger UI documentation endpoints."""
        docs_res = self.client.get("/docs")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn("swagger", docs_res.text.lower())

        schema_res = self.client.get("/openapi.json")
        self.assertEqual(schema_res.status_code, 200)
        schema = schema_res.json()
        self.assertIn("/api/fires", schema["paths"])
        self.assertIn("/api/fires/{detection_id}", schema["paths"])
        self.assertIn("/api/facilities", schema["paths"])
        self.assertIn("/api/hotspots", schema["paths"])
        self.assertIn("/api/stats", schema["paths"])
        self.assertIn("/api/classify", schema["paths"])

if __name__ == "__main__":
    unittest.main()
