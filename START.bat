@echo off
rem Double-click (or run .\START.bat in a terminal) to put the project live.
rem Keep the window open: it repairs the public link by itself.
title Guardrail Gateway - keep this window open
cd /d "%~dp0"

docker info >nul 2>&1 && goto docker_ready
echo Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
set /a tries=0
:wait_docker
"%SystemRoot%\System32\timeout.exe" /t 5 /nobreak >nul
docker info >nul 2>&1 && goto docker_ready
set /a tries+=1
if %tries% lss 60 goto wait_docker
echo Docker Desktop did not start. Open it from the Start menu, then run START.bat again.
pause
exit /b 1

:docker_ready
if not exist "C:\Program Files\Git\bin\bash.exe" (
  echo Git for Windows is missing. Install it from https://git-scm.com and run START.bat again.
  pause
  exit /b 1
)
"C:\Program Files\Git\bin\bash.exe" scripts/go_live.sh --watch
echo.
echo Auto-repair has stopped. Your site stays up until the tunnel drops.
pause
