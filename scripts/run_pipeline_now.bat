@echo off
REM ==============================================================================
REM   Trigger single execution of Part 5.1 Automated Data Pipeline on demand
REM ==============================================================================
setlocal
cd /d "%~dp0.."
echo [SIH-2026] Executing Part 5.1 Automated Pipeline (Single Run)...
python main.py --part 5 --run-once
pause
