@echo off
cd /d C:\SmartNougat\new_nougat-main\src
set PYTHONPATH=C:\SmartNougat\new_nougat-main\src;%PYTHONPATH%
echo PYTHONPATH includes: %PYTHONPATH%
echo.
echo Testing SmartNougat with page 8...
python smartnougat_0715.py "C:\test\1_AI.docx" -p 8