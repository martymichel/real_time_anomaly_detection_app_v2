@echo off
REM ============================================================
REM Start Anomaly Detection App (PySide6)
REM ============================================================

echo.
echo ============================================================
echo  IDS Camera Anomaly Detection App (PySide6)
echo ============================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Activate venv and run app
echo [INFO] Starting application...
echo.

.venv\Scripts\python.exe anomaly_detection_app_qt.py

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [INFO] Application closed successfully.
) else (
    echo.
    echo [ERROR] Application crashed with exit code: %ERRORLEVEL%
    echo.
    if %ERRORLEVEL% EQU -1073741819 (
        echo This error indicates a memory access violation.
        echo Possible causes:
        echo   - Camera initialization failed
        echo   - IDS Peak SDK issue
        echo   - Camera hardware problem
    )
    echo.
    pause
)
