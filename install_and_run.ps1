$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".vivyroom_installed")) {
    Write-Host "[Vivyroom] First-time setup on Windows: checking Python & numpy..." -ForegroundColor Cyan
    python -m pip install --quiet -r requirements.txt
    Set-Content -Path ".vivyroom_installed" -Value "installed=1"
}

Write-Host "[Vivyroom] Starting Autonomous OranPi-PocketFlow Single Chat Control Room..." -ForegroundColor Green
python -m vivyroom $args
