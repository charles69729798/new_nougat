@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%
python src\smartnougat_gui.py %*
pause