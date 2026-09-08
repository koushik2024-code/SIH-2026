# ==============================================================================
#   SIH 2026 - NTRO Challenge: Industrial Fire Detection System
#   PowerShell Windows Task Scheduler Setup (Part 5.1 Automated Pipeline)
#   Schedules the pipeline to execute automatically every 6 hours.
# ==============================================================================

$TaskName = "SIH2026_FirePipeline"
$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  SIH 2026: Setting Up 6-Hour Automated Pipeline Task Scheduler" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "Project Directory: $ProjectDir"

# Identify Python Executable
$PythonCmd = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonCmd -and (Test-Path "$ProjectDir\venv\Scripts\python.exe")) {
    $PythonCmd = "$ProjectDir\venv\Scripts\python.exe"
}

if (-not $PythonCmd) {
    Write-Error "Python executable could not be found. Please ensure Python is installed and in PATH."
    Exit 1
}

Write-Host "Detected Python: $PythonCmd" -ForegroundColor Green

$Arguments = "`"$ProjectDir\main.py`" --part 5 --run-once"

# Create or Update Task using schtasks
$Command = "schtasks /create /tn `"$TaskName`" /tr `"`"$PythonCmd`" $Arguments`" /sc hourly /mo 6 /f"
Invoke-Expression $Command

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host "Trigger interval: Every 6 hours."
    Write-Host ""
    Write-Host "Management Commands:" -ForegroundColor Yellow
    Write-Host "  - Trigger Now:   schtasks /run /tn `"$TaskName`""
    Write-Host "  - Check Status:  schtasks /query /tn `"$TaskName`""
    Write-Host "  - Remove Task:   schtasks /delete /tn `"$TaskName`" /f"
} else {
    Write-Host "[WARNING] Could not register task automatically. Try running PowerShell as Administrator." -ForegroundColor Red
}
