@echo off
cd /d "%~dp0.."
python -m app.catalog_cli --source openrouter --limit 30
pause
