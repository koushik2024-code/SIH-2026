"""
Unit Test Suite for Part 5.5: Deployment, Containerization & Health Monitoring
Tests Dockerfile, Docker Compose, Centralized Configuration, Health Checks,
Deep Diagnostics, REST endpoints, and CI/CD workflow validity.
"""

import os
import sys
import unittest
import subprocess
from pathlib import Path
import yaml
from fastapi.testclient import TestClient

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.config import AppConfig, get_config
from src.monitoring.health import SystemHealthManager
from src.api.app import app as fastapi_app
from src.web.app import create_app as create_flask_app
from scripts.deploy import run_preflight


class TestPart55Deployment(unittest.TestCase):
    """Test suite covering Part 5.5 deployment and operational capabilities."""

    def setUp(self):
        self.root_dir = ROOT_DIR
        self.fastapi_client = TestClient(fastapi_app)
        self.flask_app = create_flask_app()
        self.flask_client = self.flask_app.test_client()

    def test_dockerfile_structure(self):
        """Validate production Dockerfile instructions and security compliance."""
        dockerfile_path = self.root_dir / "Dockerfile"
        self.assertTrue(dockerfile_path.exists(), "Dockerfile must exist at repository root")

        content = dockerfile_path.read_text(encoding="utf-8")

        # Base image
        self.assertIn("FROM python:3.10-slim", content)

        # Security: Non-root user creation and switch
        self.assertIn("useradd", content)
        self.assertIn("USER appuser", content)

        # Runtime directories
        self.assertIn("/app/data", content)
        self.assertIn("/app/output", content)
        self.assertIn("/app/models", content)

        # Healthcheck probe
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("scripts/healthcheck.py", content)

        # Exposed ports
        self.assertIn("EXPOSE 8000 5000", content)

    def test_docker_compose_validity(self):
        """Validate docker-compose.yml YAML syntax, service topologies, and volumes."""
        compose_path = self.root_dir / "docker-compose.yml"
        self.assertTrue(compose_path.exists(), "docker-compose.yml must exist at repository root")

        with open(compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("services", data)
        services = data["services"]

        # Ensure all 3 required services exist
        self.assertIn("api", services)
        self.assertIn("web", services)
        self.assertIn("worker", services)

        # Verify API service
        api = services["api"]
        self.assertIn("8000:8000", api.get("ports", []))
        self.assertIn("healthcheck", api)

        # Verify Web service
        web = services["web"]
        self.assertIn("5000:5000", web.get("ports", []))
        self.assertIn("depends_on", web)

        # Verify shared volumes
        self.assertIn("volumes", data)
        volumes = data["volumes"]
        self.assertIn("fire_data", volumes)
        self.assertIn("fire_output", volumes)
        self.assertIn("fire_models", volumes)

        # Verify network
        self.assertIn("networks", data)
        self.assertIn("fire_monitoring_net", data["networks"])

    def test_docker_compose_dev(self):
        """Validate docker-compose.dev.yml development overrides."""
        dev_compose_path = self.root_dir / "docker-compose.dev.yml"
        self.assertTrue(dev_compose_path.exists(), "docker-compose.dev.yml must exist")

        with open(dev_compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("services", data)
        self.assertIn("api", data["services"])
        self.assertIn("web", data["services"])

    def test_dockerignore_file(self):
        """.dockerignore should ignore caches, virtualenvs, and test temporary files."""
        dockerignore_path = self.root_dir / ".dockerignore"
        self.assertTrue(dockerignore_path.exists(), ".dockerignore must exist")
        content = dockerignore_path.read_text(encoding="utf-8")
        self.assertIn("__pycache__", content)
        self.assertIn(".git", content)

    def test_app_config_loading_and_defaults(self):
        """Test AppConfig initialization and sensible defaults."""
        config = get_config()
        self.assertIsInstance(config, AppConfig)
        self.assertIn(config.ENV, ["development", "production", "testing"])
        self.assertEqual(config.API_PORT, 8000)
        self.assertEqual(config.WEB_PORT, 5000)
        self.assertTrue(config.DATA_DIR.exists())
        self.assertTrue(config.OUTPUT_DIR.exists())

    def test_environment_validation(self):
        """Test AppConfig.validate_environment() pre-flight sanity checks."""
        config = get_config()
        result = config.validate_environment()
        self.assertIn("is_valid", result)
        self.assertIn("errors", result)
        self.assertIn("warnings", result)
        self.assertIn("details", result)
        self.assertIn("directories", result["details"])
        self.assertIn("ports", result["details"])

    def test_system_health_manager_probes(self):
        """Test SystemHealthManager liveness, readiness, and deep diagnostics."""
        mgr = SystemHealthManager()

        # 1. Liveness
        liveness = mgr.get_liveness()
        self.assertEqual(liveness["status"], "alive")
        self.assertIn("uptime_seconds", liveness)
        self.assertGreaterEqual(liveness["uptime_seconds"], 0)

        # 2. Readiness
        readiness = mgr.get_readiness()
        self.assertIn(readiness["status"], ["ready", "not_ready"])
        self.assertIn("database_accessible", readiness["components"])

        # 3. Deep diagnostics
        deep = mgr.get_deep_diagnostics()
        self.assertIn("status", deep)
        self.assertIn("database", deep)
        self.assertIn("storage", deep)
        self.assertIn("ml_model", deep)
        self.assertIn("telemetry", deep)
        self.assertIn("alerts", deep)
        self.assertIn("system", deep)

    def test_standalone_healthcheck_script(self):
        """Test scripts/healthcheck.py execution with --service system."""
        script_path = self.root_dir / "scripts" / "healthcheck.py"
        self.assertTrue(script_path.exists(), "scripts/healthcheck.py must exist")

        res = subprocess.run(
            [sys.executable, str(script_path), "--service", "system"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Healthcheck failed: {res.stderr}")
        self.assertIn("[OK]", res.stdout)

    def test_fastapi_deep_health_endpoint(self):
        """Test FastAPI /health and /api/health/deep endpoints."""
        # Standard health
        resp_std = self.fastapi_client.get("/health")
        self.assertEqual(resp_std.status_code, 200)

        # Root deep health
        resp_deep = self.fastapi_client.get("/health/deep")
        self.assertEqual(resp_deep.status_code, 200)
        data = resp_deep.json()
        self.assertIn("database", data)
        self.assertIn("storage", data)

        # API prefix deep health
        resp_api_deep = self.fastapi_client.get("/api/health/deep")
        self.assertEqual(resp_api_deep.status_code, 200)

    def test_flask_health_endpoints(self):
        """Test Flask /health and /api/health/deep endpoints."""
        resp_health = self.flask_client.get("/health")
        self.assertEqual(resp_health.status_code, 200)
        health_json = resp_health.get_json()
        self.assertEqual(health_json.get("status"), "ready")

        resp_deep = self.flask_client.get("/api/health/deep")
        self.assertEqual(resp_deep.status_code, 200)
        deep_json = resp_deep.get_json()
        self.assertIn("status", deep_json)
        self.assertIn("database", deep_json)

    def test_ci_cd_workflow_validity(self):
        """Validate .github/workflows/ci_cd.yml GitHub Actions workflow."""
        workflow_path = self.root_dir / ".github" / "workflows" / "ci_cd.yml"
        self.assertTrue(workflow_path.exists(), "ci_cd.yml must exist")

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        self.assertIn("jobs", workflow)
        jobs = workflow["jobs"]
        self.assertIn("lint-and-validate", jobs)
        self.assertIn("test-suite", jobs)
        self.assertIn("docker-build-verify", jobs)
        self.assertIn("deployment-readiness", jobs)

    def test_deploy_preflight_runner(self):
        """Test scripts/deploy.py run_preflight() returns True."""
        result = run_preflight()
        self.assertTrue(result, "Deployment preflight check should pass successfully")


if __name__ == "__main__":
    unittest.main()
