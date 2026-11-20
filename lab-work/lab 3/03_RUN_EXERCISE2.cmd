@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" src\exercise2.py --cpu
if errorlevel 1 (
  echo FAILED. Read the error above.
) else (
  echo COMPLETED.
)
pause
