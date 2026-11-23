@echo off
rem Double-click (or run .\STOP.bat in a terminal) to take the project offline.
cd /d "%~dp0"
docker compose -f docker-compose.local.yml --profile public down
echo.
echo Stopped. Your public link now shows "Model offline".
echo Run START.bat to go live again.
pause
