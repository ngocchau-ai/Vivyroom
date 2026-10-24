@echo off
setlocal
cd /d "%~dp0"

if not exist ".vivyroom_installed" (
    echo [Vivyroom] First-time setup on Windows: checking Python & numpy...
    python -m pip install --quiet -r requirements.txt
    echo installed=1 > .vivyroom_installed
)

echo [Vivyroom] Starting Autonomous OranPi-PocketFlow Single Chat Control Room...
python -m vivyroom %*
