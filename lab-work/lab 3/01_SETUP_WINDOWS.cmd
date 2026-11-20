@echo off
cd /d "%~dp0"
py -3.14 setup_env.py
if errorlevel 1 (
  echo FAILED. Read the error above.
) else (
  echo COMPLETED.
)
pause
