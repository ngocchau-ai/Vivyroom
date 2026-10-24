Set-Location -Path $PSScriptRoot
Write-Host "[Vivyroom] Starting Autonomous OranPi-PocketFlow Single Chat Control Room..." -ForegroundColor Green
python -m vivyroom $args
