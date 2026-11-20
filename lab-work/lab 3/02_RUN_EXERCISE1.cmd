@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" src\exercise1.py
if errorlevel 1 (
  echo FAILED. Read the error above.
) else (
  echo COMPLETED.
)
pause
