@echo off
echo ========================================
echo Download AI Models for SmartNougat
echo ========================================
echo.

cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Set PYTHONPATH
set PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%

:: Run model download
echo Downloading models...
echo This requires internet connection
echo File size: ~1.5GB
echo.
python scripts\setup_models.py

echo.
echo ========================================
echo Model download complete!
echo ========================================
echo.
echo You can now run SmartNougat offline.
echo.
pause