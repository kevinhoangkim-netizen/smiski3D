@echo off
setlocal
cd /d "%~dp0"
title Little Figures Studio
py -3.12 --version >nul 2>&1
if not errorlevel 1 goto use_launcher
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" goto use_local
if exist "%ProgramFiles%\Python312\python.exe" goto use_system

echo Little Figures Studio needs Python 3.12 installed once.
echo Opening the official Python Windows installer download.
echo Complete the installation, then double-click this file again.
echo No Node.js, hosting account, or coding is required.
start "" "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
pause
exit /b 1

:use_launcher
py -3.12 scripts\launch_local.py
goto finished

:use_local
"%LocalAppData%\Programs\Python\Python312\python.exe" scripts\launch_local.py
goto finished

:use_system
"%ProgramFiles%\Python312\python.exe" scripts\launch_local.py

:finished
echo.
echo This studio session has ended. Open this file again to restart.
pause
