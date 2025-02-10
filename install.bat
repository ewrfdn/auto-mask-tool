@echo off
REM Check and auto-elevate to admin rights
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

setlocal EnableDelayedExpansion

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Please run as Administrator!
    pause
    exit /b 1
)

REM Get current directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Get current PATH
for /f "tokens=2*" %%a in ('reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH') do set "CURRENT_PATH=%%b"

REM Check if path already exists
echo !CURRENT_PATH! | find /i "%SCRIPT_DIR%" > nul
if !errorLevel! equ 0 (
    echo Directory already in PATH
) else (
    REM Add to PATH
    setx /M PATH "%CURRENT_PATH%;%SCRIPT_DIR%"
    if !errorLevel! equ 0 (
        echo Successfully added to PATH
    ) else (
        echo Failed to add to PATH
    )
)

REM Set AUTOCUT_ENV_PATH environment variable
setx /M AUTOCUT_ENV_PATH "%SCRIPT_DIR%\autocut"
if !errorLevel! equ 0 (
    echo Successfully set AUTOCUT_ENV_PATH
) else (
    echo Failed to set AUTOCUT_ENV_PATH
)

echo.
echo Please restart your command prompt to use autocut from anywhere.
pause