@echo off
chcp 65001 >nul
cls
echo ============================================
echo   SmartNougat - Complete Installation
echo ============================================
echo.

cd /d "%~dp0"

echo [1/6] Installing Python packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [2/6] Downloading nougat-latex-ocr...
if not exist "nougat-latex-ocr" (
    git clone https://github.com/NormXU/nougat-latex-ocr.git
) else (
    echo nougat-latex-ocr already exists, skipping...
)

echo.
echo [3/6] Downloading MathJax for offline use...
python download_mathjax.py

echo.
echo [4/6] Creating model directories...
if not exist "models" mkdir models

echo.
echo [5/6] Downloading AI models...
echo This may take a while (models are large)...
python download_norm_nougat_model.py
python download_real_yolo_model.py

echo.
echo [6/6] Testing installation...
python test_dependencies.py

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To run SmartNougat:
echo - GUI: run_smartnougat_gui_v2.bat
echo - CLI: run_smartnougat_cli.bat "file.pdf" -p 1
echo.
echo Example: run_smartnougat_cli.bat "C:\test\document.pdf" -p 1-5
echo.
pause