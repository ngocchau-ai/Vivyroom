@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0
echo ==========================================================================
echo  Launching Vivyroom (Portable ^<2GB RAM Unified Vivy + Cautreo Cockpit)
echo ==========================================================================
python -m vivyroom.server %*
