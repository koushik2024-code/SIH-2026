#!/usr/bin/env bash
# ==============================================================================
#   SIH 2026 - NTRO Challenge: Industrial Fire Detection System
#   Cron Job Registration Script (Part 5.1 Automated Pipeline)
#   Runs every 6 hours: 0 */6 * * *
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==============================================================="
echo "  Setting up 6-Hour Cron Job for SIH 2026 Pipeline"
echo "==============================================================="
echo "Project Directory: $PROJECT_DIR"

PYTHON_EXE=""
if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="$(command -v python3)"
elif command -v python &>/dev/null; then
    PYTHON_EXE="$(command -v python)"
else
    echo "[ERROR] Python executable not found."
    exit 1
fi

echo "Using Python: $PYTHON_EXE"

LOG_FILE="$PROJECT_DIR/data/pipeline_cron.log"
CRON_CMD="0 */6 * * * cd $PROJECT_DIR && $PYTHON_EXE main.py --part 5 --run-once >> $LOG_FILE 2>&1"

# Check if entry already exists in crontab
if crontab -l 2>/dev/null | grep -F "SIH-2026" >/dev/null; then
    echo "[INFO] SIH-2026 pipeline cron job is already installed in crontab."
else
    (crontab -l 2>/dev/null; echo "# SIH-2026 6-Hour Automated Fire Pipeline"; echo "$CRON_CMD") | crontab -
    echo "[SUCCESS] Cron job registered successfully!"
    echo "Cron schedule: 0 */6 * * * (Every 6 hours)"
    echo "Logs destination: $LOG_FILE"
fi
