@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
py -3.14 -c "import sys; assert sys.version_info[:2] == (3,14)" >nul 2>&1
if not errorlevel 1 goto tagged
py -c "import sys; assert sys.version_info[:2] == (3,14)" >nul 2>&1
if not errorlevel 1 goto defaultpy
python -c "import sys; assert sys.version_info[:2] == (3,14)" >nul 2>&1
if not errorlevel 1 goto direct
 echo Khong tim thay Python 3.14. Chay py --list de kiem tra.
goto failed
:tagged
py -3.14 setup_env.py
if errorlevel 1 goto failed
goto done
:defaultpy
py setup_env.py
if errorlevel 1 goto failed
goto done
:direct
python setup_env.py
if errorlevel 1 goto failed
:done
echo SETUP OK. Chay 02_RUN_EXERCISE1.cmd.
pause
exit /b 0
:failed
echo CAI DAT THAT BAI. Hay chup phan loi ben tren.
pause
exit /b 1
