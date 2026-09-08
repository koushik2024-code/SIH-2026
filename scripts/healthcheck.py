#!/usr/bin/env python3
"""
Standalone Healthcheck CLI & Container Probe
Part 5.5: Deployment, Containerization & Health Monitoring

Executes zero-dependency health inspection suitable for Docker HEALTHCHECK
and orchestrator readiness probes.
"""

import sys
import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_http_endpoint(url: str, timeout: float = 5.0) -> bool:
    """Check if an HTTP endpoint returns a 200 OK status code."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SIH-2026-HealthCheck/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                print(f"[OK] Health probe succeeded: {url} -> HTTP 200")
                return True
            else:
                print(f"[FAIL] Unexpected status code from {url}: {response.status}", file=sys.stderr)
                return False
    except urllib.error.HTTPError as e:
        print(f"[FAIL] HTTP error from {url}: {e.code} {e.reason}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"[FAIL] Connection failed to {url}: {e.reason}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[FAIL] Probe error for {url}: {e}", file=sys.stderr)
        return False


def check_system_direct() -> bool:
    """Directly inspect system readiness via Python module without network."""
    try:
        from src.monitoring.health import SystemHealthManager
        manager = SystemHealthManager()
        readiness = manager.get_readiness()
        if readiness.get("status") == "ready":
            print(f"[OK] Direct system healthcheck passed: {readiness}")
            return True
        else:
            print(f"[FAIL] Direct system healthcheck not ready: {readiness}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[FAIL] Direct healthcheck encountered exception: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Fire Monitoring Healthcheck Utility")
    parser.add_argument(
        "--service",
        choices=["api", "web", "system"],
        default="api",
        help="Target service to probe (default: api)",
    )
    parser.add_argument(
        "--host",
        default="http://localhost",
        help="Host address (default: http://localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Target port (default: 8000 for api, 5000 for web)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Probe timeout in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    if args.service == "system":
        success = check_system_direct()
        sys.exit(0 if success else 1)

    port = args.port
    if port is None:
        port = 8000 if args.service == "api" else 5000

    url = f"{args.host}:{port}/health"
    success = check_http_endpoint(url, timeout=args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
