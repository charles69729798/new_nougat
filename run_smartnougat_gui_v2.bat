@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%;%CD%\src;%CD%\nougat-latex-ocr;%PYTHONPATH%
echo Starting SmartNougat GUI v2...
python src\smartnougat_gui_0717_v2.py %*
pause