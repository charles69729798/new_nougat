@echo off 
cd /d "%~dp0" 
call venv\Scripts\activate.bat 
set PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH% 
python -m src.smartnougat_gui %* 
