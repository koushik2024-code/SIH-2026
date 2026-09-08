#!/usr/bin/env python3
"""
Deployment Automation & Management Utility
Part 5.5: Deployment, Containerization & Health Monitoring

Provides unified CLI commands for pre-flight validation, Docker container lifecycle,
and live health diagnostics.
"""

import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_config
from src.monitoring.health import SystemHealthManager


def run_preflight() -> bool:
    """Run comprehensive deployment pre-flight verification."""
    config = get_config()
    validation = config.validate_environment()
    health_mgr = SystemHealthManager()
    readiness = health_mgr.get_readiness()

    print("===============================================================")
    print("   SIH 2026: PRODUCTION DEPLOYMENT PRE-FLIGHT VERIFICATION    ")
    print("===============================================================")
    print(f"  [+] Environment:          {config.ENV.upper()}")
    print(f"  [+] Debug Mode:           {config.DEBUG}")
    print(f"  [+] API Target:           {config.API_HOST}:{config.API_PORT}")
    print(f"  [+] Web Target:           {config.WEB_HOST}:{config.WEB_PORT}")
    print(f"  [+] SQLite Database:      {config.DATABASE_PATH} ({'EXISTS' if config.DATABASE_PATH.exists() else 'NOT INITIALIZED'})")
    print(f"  [+] Machine Learning:     {config.MODEL_PATH} ({'LOADED' if config.MODEL_PATH.exists() else 'HEURISTIC FALLBACK'})")
    print(f"  [+] System Readiness:     {readiness.get('status', 'unknown').upper()}")
    print("---------------------------------------------------------------")

    # Docker presence check
    docker_installed = shutil.which("docker") is not None
    compose_installed = False
    if docker_installed:
        try:
            res = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
            compose_installed = res.returncode == 0
        except Exception:
            compose_installed = False

    print(f"  [+] Docker Installed:     {'YES' if docker_installed else 'NO (Install Docker to run in container)'}")
    print(f"  [+] Docker Compose:       {'YES' if compose_installed else 'NO'}")
    print("---------------------------------------------------------------")

    if validation["errors"]:
        print("  [!] PRE-FLIGHT ERRORS:")
        for err in validation["errors"]:
            print(f"      - {err}")
    else:
        print("  [+] Directory Storage:    ALL REQUIRED DIRECTORIES WRITABLE")

    if validation["warnings"]:
        print("  [*] PRE-FLIGHT WARNINGS:")
        for warn in validation["warnings"]:
            print(f"      - {warn}")

    print("===============================================================")
    if validation["is_valid"]:
        print("  >>> STATUS: SYSTEM READY FOR PRODUCTION DEPLOYMENT <<<      ")
    else:
        print("  >>> STATUS: PRE-FLIGHT FAILED - RESOLVE ERRORS ABOVE <<<    ")
    print("===============================================================")

    return validation["is_valid"]


def main():
    parser = argparse.ArgumentParser(description="Fire Monitoring Deployment Utility")
    parser.add_argument(
        "action",
        choices=["preflight", "build", "up", "down", "status", "logs"],
        default="preflight",
        nargs="?",
        help="Deployment action to execute (default: preflight)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use docker-compose.dev.yml for local development",
    )

    args = parser.parse_args()

    if args.action == "preflight":
        success = run_preflight()
        sys.exit(0 if success else 1)

    compose_file = "docker-compose.dev.yml" if args.dev else "docker-compose.yml"

    if args.action == "build":
        cmd = ["docker", "compose", "-f", compose_file, "build"]
    elif args.action == "up":
        cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
    elif args.action == "down":
        cmd = ["docker", "compose", "-f", compose_file, "down"]
    elif args.action == "logs":
        cmd = ["docker", "compose", "-f", compose_file, "logs", "-f"]
    elif args.action == "status":
        cmd = ["docker", "compose", "-f", compose_file, "ps"]
    else:
        cmd = ["docker", "compose", "-f", compose_file, "ps"]

    try:
        res = subprocess.run(cmd)
        sys.exit(res.returncode)
    except FileNotFoundError:
        print("[ERROR] Docker or docker-compose is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
