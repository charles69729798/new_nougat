@echo off
chcp 65001 > nul
title SmartNougat GUI Batch - 폴더 일괄 처리

echo SmartNougat GUI Batch 시작 중...
echo.

cd /d "%~dp0"

python src\smartnougat_gui_batch.py

pause