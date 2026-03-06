@echo off
echo =========================================
echo Building PROJECT ARGUS - Claims Scanner
echo =========================================

REM Check if PyInstaller is installed
python -m PyInstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)
)

echo.
echo Installing project dependencies from requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies!
    pause
    exit /b %errorlevel%
)

echo.
echo [1/2] Building claims_scanner.exe...
python -m PyInstaller claims_scanner.spec --noconfirm

if %errorlevel% neq 0 (
    echo Error building claims_scanner.exe!
    pause
    exit /b %errorlevel%
)

echo.
echo [2/2] Building argus_updater.exe...
python -m PyInstaller argus_updater.spec --noconfirm

if %errorlevel% neq 0 (
    echo Error building argus_updater.exe!
    pause
    exit /b %errorlevel%
)

echo.
echo =========================================
echo Build Complete!  Executables are in:
echo   dist\claims_scanner.exe
echo   dist\argus_updater.exe
echo =========================================
pause
