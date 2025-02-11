@echo off
setlocal
set ENV_NAME=autocut

REM Check if AUTOCUT_ENV_PATH environment variable exists
if defined AUTOCUT_ENV_PATH (
    set "ENV_PATH=%AUTOCUT_ENV_PATH%"
) else (
    set "ENV_PATH=.\%ENV_NAME%"
)

REM Check if virtual environment exists
if not exist "%ENV_PATH%" (
    echo Error: Virtual environment not found at %ENV_PATH%!
    echo Please run 'init.cmd' first to set up the environment.
    exit /b 1
)

REM Activate virtual environment
call "%ENV_PATH%\Scripts\activate.bat"

REM Run the main script

python "%ENV_PATH%\..\src\autocut.py" %*

REM Deactivate virtual environment
deactivate

endlocal
