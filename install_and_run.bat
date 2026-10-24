@echo off
setlocal
cd /d "%~dp0"
echo [Vivyroom] Starting Autonomous OranPi-PocketFlow Single Chat Control Room...
python -m vivyroom %*
