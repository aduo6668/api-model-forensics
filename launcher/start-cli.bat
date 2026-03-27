@echo off
cd /d %~dp0..
if "%~1"=="" (
    cmd /k python -m app.cli --help
) else (
    python -m app.cli %*
)
