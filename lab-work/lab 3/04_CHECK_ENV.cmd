@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" check_env.py
if errorlevel 1 (
  echo FAILED. Read the error above.
) else (
  echo COMPLETED.
)
pause
