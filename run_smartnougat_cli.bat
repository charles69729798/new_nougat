@echo off
cd /d "%~dp0\src"
echo SmartNougat CLI
echo.
echo Usage: run_smartnougat_cli.bat [input_file] [options]
echo Options:
echo   -p [pages]    : Page range (e.g., 1-5 or 1,3,5)
echo   -o [output]   : Output directory
echo   --debug       : Debug mode
echo.

if "%1"=="" (
    echo Error: Please provide input file
    echo Example: run_smartnougat_cli.bat "C:\test\1_AI.docx" -p 4
    pause
    exit /b 1
)

python smartnougat_0715.py %*
pause