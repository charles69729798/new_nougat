@echo off
echo ========================================
echo Setup Downloaded Models
echo ========================================
echo.

cd /d "%~dp0"

:: Check if model files exist
if not exist "models\pytorch_model.bin" (
    echo ERROR: pytorch_model.bin not found in models folder!
    echo Please ensure all model files are in: %CD%\models
    pause
    exit /b 1
)

echo Found model files in models folder
echo.

:: Create proper cache structure
echo Creating cache structure...

:: Option 1: Local cache in project
set "CACHE_DIR=models\.cache\models--Norm--nougat-latex-base\snapshots\3adffd2e59fd34f32eec3be36eab998b3a7b17b2"
mkdir "%CACHE_DIR%" 2>nul

:: Copy all files
echo Copying model files to cache...
copy /Y "models\*.json" "%CACHE_DIR%\" >nul
copy /Y "models\pytorch_model.bin" "%CACHE_DIR%\" >nul

:: Create refs
mkdir "models\.cache\models--Norm--nougat-latex-base\refs" 2>nul
echo main > "models\.cache\models--Norm--nougat-latex-base\refs\main"
echo 3adffd2e59fd34f32eec3be36eab998b3a7b17b2 >> "models\.cache\models--Norm--nougat-latex-base\refs\main"

:: Option 2: Also copy to user's HF cache (optional)
set "HF_CACHE=%USERPROFILE%\.cache\huggingface\hub\models--Norm--nougat-latex-base\snapshots\3adffd2e59fd34f32eec3be36eab998b3a7b17b2"
mkdir "%HF_CACHE%" 2>nul

echo.
echo Also copying to Hugging Face cache...
copy /Y "models\*.json" "%HF_CACHE%\" >nul
copy /Y "models\pytorch_model.bin" "%HF_CACHE%\" >nul

:: Create HF refs
mkdir "%USERPROFILE%\.cache\huggingface\hub\models--Norm--nougat-latex-base\refs" 2>nul
echo main > "%USERPROFILE%\.cache\huggingface\hub\models--Norm--nougat-latex-base\refs\main"
echo 3adffd2e59fd34f32eec3be36eab998b3a7b17b2 >> "%USERPROFILE%\.cache\huggingface\hub\models--Norm--nougat-latex-base\refs\main"

:: Set environment variables for the session
echo.
echo Setting environment variables...
set TRANSFORMERS_CACHE=%CD%\models\.cache
set HF_HOME=%CD%\models\.cache
set HUGGINGFACE_HUB_CACHE=%CD%\models\.cache

:: Verify installation
echo.
echo ========================================
echo Verifying installation...
echo ========================================
echo.

if exist "%CACHE_DIR%\pytorch_model.bin" (
    echo ✓ Model files copied to local cache
    echo   Location: %CACHE_DIR%
) else (
    echo ✗ Failed to copy to local cache
)

if exist "%HF_CACHE%\pytorch_model.bin" (
    echo ✓ Model files copied to HF cache
    echo   Location: %HF_CACHE%
) else (
    echo ✗ Failed to copy to HF cache (optional)
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run SmartNougat offline!
echo Run: run_smartnougat.bat
echo.
pause