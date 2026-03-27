@echo off
cd /d "%~dp0.."
python -m app.cli --emit-chat-self-test-pack --format text
pause
