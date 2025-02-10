@echo off
setlocal

set ENV_NAME=autocut
set PYTHON_VERSION=3.12
set ENV_PATH=.\%ENV_NAME%

REM Check architecture
wmic os get osarchitecture | findstr "ARM" > nul
if %ERRORLEVEL% == 0 (
    echo Detected ARM64 architecture
    set PYTHON_INSTALLER=python-%PYTHON_VERSION%.0-arm64.exe
    set ARCH_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%.0/python-%PYTHON_VERSION%.0-arm64.exe
) else (
    echo Detected x64 architecture
    set PYTHON_INSTALLER=python-%PYTHON_VERSION%.0-amd64.exe
    set ARCH_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%.0/python-%PYTHON_VERSION%.0-amd64.exe
)

echo Checking Python %PYTHON_VERSION%...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found, installing Python %PYTHON_VERSION%...
    curl -o %PYTHON_INSTALLER% %ARCH_URL%
    %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1
    del %PYTHON_INSTALLER%
)

REM Check if venv exists
if exist "%ENV_PATH%" (
    echo Virtual environment exists, activating...
) else (
    echo Creating virtual environment '%ENV_NAME%'...
    python -m venv %ENV_PATH%
)

REM Activate venv and update pip
call "%ENV_PATH%\Scripts\activate.bat"
python -m pip install --upgrade pip

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo.
echo Usage:
echo 1. Activate: %ENV_PATH%\Scripts\activate.bat
echo 2. Deactivate: deactivate
echo 3. Environment path: %ENV_PATH%

endlocal
