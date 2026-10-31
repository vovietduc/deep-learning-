@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
".venv314\Scripts\python.exe" main.py --exercise 2
if errorlevel 1 goto failed
echo.
echo HOAN TAT.
pause
exit /b 0
:failed
echo.
echo CO LOI. Hay chup phan loi ben tren de kiem tra.
pause
exit /b 1
