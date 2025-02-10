@echo off
setlocal

set ENV_NAME=autocut
set ENV_PATH=.\%ENV_NAME%

REM Check if virtual environment exists
if not exist "%ENV_PATH%" (
    echo Error: Virtual environment not found!
    echo Please run 'init.cmd' first to set up the environment.
    exit /b 1
)

REM Activate virtual environment
call "%ENV_PATH%\Scripts\activate.bat"

REM Run the main script
echo Starting AutoCut...
python ./src/autocut.py %*

REM Deactivate virtual environment
deactivate

endlocal
