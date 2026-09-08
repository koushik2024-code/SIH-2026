@echo off
REM ==============================================================================
REM   SIH 2026 - NTRO Challenge: Industrial Fire Detection System
REM   Windows Task Scheduler Registration Script (Part 5.1 Automated Pipeline)
REM   Schedules the pipeline to execute automatically every 6 hours.
REM ==============================================================================

setlocal enabledelayedexpansion

set TASK_NAME=SIH2026_FirePipeline
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."
set PROJECT_DIR=%CD%

echo [SIH-2026] Registering Windows Scheduled Task for 6-Hour Automated Pipeline...
echo Project Directory: %PROJECT_DIR%

REM Detect Python executable
where python >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where python') do set PYTHON_EXE=%%i & goto :found_python
)

REM Fallback if virtualenv exists
if exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    set PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe
    goto :found_python
)

echo [ERROR] Could not detect python.exe. Please ensure Python is in your PATH.
pause
exit /b 1

:found_python
echo Using Python Executable: %PYTHON_EXE%

set ACTION_CMD="\"%PYTHON_EXE%\" \"%PROJECT_DIR%\main.py\" --part 5 --run-once"

echo Creating scheduled task '%TASK_NAME%' to run every 6 hours...
schtasks /create /tn "%TASK_NAME%" /tr "%ACTION_CMD%" /sc hourly /mo 6 /f /ru "%USERNAME%"

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================================
    echo [SUCCESS] Task '%TASK_NAME%' has been successfully registered!
    echo It will execute automatically every 6 hours.
    echo.
    echo Useful commands:
    echo   - Run immediately:  schtasks /run /tn "%TASK_NAME%"
    echo   - Check status:     schtasks /query /tn "%TASK_NAME%"
    echo   - Delete task:      schtasks /delete /tn "%TASK_NAME%" /f
    echo ==============================================================================
) else (
    echo [ERROR] Failed to register task. You may need to run this command as Administrator.
)

pause
