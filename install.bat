@echo off
setlocal enabledelayedexpansion
REM Anomaly Detection App - Installation Script (using uv)
REM =======================================================
REM
REM This script will:
REM 1. Install uv (if not already installed)
REM 2. Create a Python virtual environment with uv
REM 3. Install all required dependencies
REM 4. Verify GPU/CUDA availability
REM

echo.
echo ========================================================
echo  Anomaly Detection App - Installation (uv)
echo ========================================================
echo.

REM ========================================
REM Step 1: Check/Install uv
REM ========================================

echo [INFO] Checking for uv...

REM Check if uv is in PATH
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] uv found in PATH
    uv --version
    goto :uv_ready
)

REM Check if uv is in user's .cargo/bin (common location)
if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    echo [OK] uv found in %USERPROFILE%\.cargo\bin
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    uv --version
    goto :uv_ready
)

REM Check if uv is in AppData\Local (Windows uv installer location)
if exist "%LOCALAPPDATA%\uv\uv.exe" (
    echo [OK] uv found in %LOCALAPPDATA%\uv
    set "PATH=%LOCALAPPDATA%\uv;%PATH%"
    uv --version
    goto :uv_ready
)

REM UV not found - install it
echo [WARN] uv not found - installing uv...
echo.
echo This will download and install uv via PowerShell...
echo.

REM Install uv via PowerShell
powershell -NoProfile -ExecutionPolicy ByPass -Command "& {irm https://astral.sh/uv/install.ps1 | iex}"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install uv!
    echo.
    echo Please install manually:
    echo   PowerShell: irm https://astral.sh/uv/install.ps1 ^| iex
    echo   Or visit: https://github.com/astral-sh/uv
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] UV installation completed. Checking location...

REM Try to find newly installed uv
if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    echo [OK] Found uv in %USERPROFILE%\.cargo\bin
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
) else if exist "%LOCALAPPDATA%\uv\uv.exe" (
    echo [OK] Found uv in %LOCALAPPDATA%\uv
    set "PATH=%LOCALAPPDATA%\uv;%PATH%"
) else (
    echo [ERROR] UV installed but not found in expected locations!
    echo.
    echo Please restart your terminal and run this script again.
    echo Or add uv to your PATH manually.
    echo.
    pause
    exit /b 1
)

:uv_ready
echo.
echo [OK] uv is ready:
uv --version
echo.

REM ========================================
REM Step 2: Python Version Check
REM ========================================

echo [INFO] Checking Python version...

REM uv can manage Python versions, but let's check if one exists
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] No Python found, uv will install Python 3.11...
    set "USE_UV_PYTHON=1"
) else (
    python --version
    python -c "import sys; exit(0 if sys.version_info >= (3,8) and sys.version_info < (3,13) else 1)" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARN] Python version not compatible (need 3.8-3.12)
        echo [INFO] uv will install Python 3.11...
        set "USE_UV_PYTHON=1"
    ) else (
        echo [OK] Python version compatible
        set "USE_UV_PYTHON=0"
    )
)

echo.

REM ========================================
REM Step 3: Create Virtual Environment with uv
REM ========================================

if exist ".venv\" (
    echo [INFO] Virtual environment already exists
    echo.
    set /p "recreate=Do you want to recreate it? (y/N): "
    if /i "!recreate!"=="y" (
        echo [INFO] Removing old virtual environment...
        rmdir /s /q .venv
        goto create_venv
    ) else (
        goto skip_venv
    )
)

:create_venv
echo [INFO] Creating virtual environment with uv...
echo.

REM Always use Python 3.12 for PyTorch CUDA compatibility
echo [INFO] Creating venv with Python 3.12 (PyTorch CUDA requirement)...
uv venv .venv --python 3.12

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to create virtual environment
    echo.
    pause
    exit /b 1
)

echo [OK] Virtual environment created
echo.

:skip_venv

REM ========================================
REM Step 4: Activate Virtual Environment
REM ========================================

echo [INFO] Activating virtual environment...

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv\Scripts\activate.bat not found!
    echo Virtual environment seems incomplete.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM ========================================
REM Step 5: Install PyTorch with CUDA using uv
REM ========================================

echo [INFO] Checking for existing PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} found')" 2>nul
if %errorlevel% equ 0 (
    echo [INFO] PyTorch already installed
    python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
    echo.
    set /p "reinstall_torch=Do you want to reinstall PyTorch? (y/N): "
    if /i "!reinstall_torch!"=="y" (
        goto install_torch
    ) else (
        goto skip_torch
    )
)

:install_torch
echo.
echo [INFO] Installing PyTorch with CUDA 12.1 using uv...
echo This may take several minutes (uv is very fast!)...
echo.

REM uv pip install with extra index URL
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install PyTorch
    echo.
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo [OK] PyTorch installed
echo.

:skip_torch

REM ========================================
REM Step 6: Install FAISS
REM ========================================

echo [INFO] Checking for existing FAISS installation...
python -c "import faiss; print(f'FAISS {faiss.__version__} found')" 2>nul
if %errorlevel% equ 0 (
    echo [INFO] FAISS already installed
    python -c "import faiss; res = faiss.StandardGpuResources(); print('  GPU support: Available'); del res" 2>nul && echo   GPU version: Yes || echo   GPU version: No (CPU only)
    echo.
    set /p "reinstall_faiss=Do you want to reinstall FAISS? (y/N): "
    if /i "!reinstall_faiss!"=="y" (
        goto install_faiss
    ) else (
        goto skip_faiss
    )
)

:install_faiss
echo.
echo [INFO] Installing FAISS...
echo.
echo NOTE: FAISS-GPU is not available for Windows via PyPI.
echo The app will use FAISS-CPU + PyTorch GPU fallback (torch.cdist).
echo This provides excellent performance on modern GPUs!
echo.

uv pip install faiss-cpu

if %errorlevel% neq 0 (
    echo [WARN] FAISS installation failed
    echo [INFO] App will use PyTorch's torch.cdist (GPU-accelerated)
    echo.
) else (
    echo [OK] FAISS-CPU installed
    echo.
)

:skip_faiss

REM ========================================
REM Step 7: Install Other Dependencies
REM ========================================

echo [INFO] Installing other dependencies using uv...
echo.

if not exist "requirements_qt.txt" (
    echo [ERROR] requirements_qt.txt not found!
    echo.
    pause
    exit /b 1
)

REM uv is MUCH faster than pip!
uv pip install -r requirements_qt.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo.
    echo Please check requirements_qt.txt and your internet connection.
    echo.
    pause
    exit /b 1
)

echo [OK] All dependencies installed
echo.

REM ========================================
REM Step 8: Verify Installation
REM ========================================

echo.
echo ========================================================
echo  Verifying Installation
echo ========================================================
echo.

echo [INFO] Checking critical dependencies...
echo.

python -c "import sys; print(f'Python: {sys.version.split()[0]}')"
echo.

python -c "import PySide6; print(f'[OK] PySide6 {PySide6.__version__}')" 2>nul || echo [FAIL] PySide6 not found

python -c "import torch; print(f'[OK] PyTorch {torch.__version__}'); print(f'     CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"})')" 2>nul || echo [FAIL] PyTorch not found

python -c "import transformers; print(f'[OK] Transformers {transformers.__version__}')" 2>nul || echo [FAIL] Transformers not found

python -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')" 2>nul || echo [FAIL] OpenCV not found

python -c "import numpy; print(f'[OK] NumPy {numpy.__version__}')" 2>nul || echo [FAIL] NumPy not found

python -c "import faiss; print(f'[OK] FAISS {faiss.__version__}'); res = faiss.StandardGpuResources(); print('     GPU: Available'); del res" 2>nul || (python -c "import faiss; print(f'[OK] FAISS {faiss.__version__} (CPU)')" 2>nul || echo [INFO] FAISS not installed - PyTorch fallback active)

echo.

REM Check IDS Peak SDK
python -c "from ids_peak import ids_peak; print('[OK] IDS Peak SDK')" 2>nul || echo [WARN] IDS Peak SDK not found (install from https://www.ids-imaging.com/downloads.html)

echo.
echo ========================================================
echo  Installation Complete!
echo ========================================================
echo.
echo Installed with uv - the blazingly fast Python package manager!
echo.
echo Next steps:
echo 1. Connect your IDS camera
echo 2. Run: start_anomaly_detection_qt.bat
echo.
echo For manual start:
echo   - Activate venv: .venv\Scripts\activate.bat
echo   - Run app: python anomaly_detection_app_qt.py
echo.

REM Display performance info
echo Performance:
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>nul
python -c "import faiss; res = faiss.StandardGpuResources(); print('  FAISS: GPU-accelerated'); del res" 2>nul || echo   FAISS: CPU (PyTorch GPU fallback enabled)

echo.
echo ========================================================
echo.

pause
