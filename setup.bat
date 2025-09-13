@echo off
REM Quick Setup Script for Ceramic Armor ML API (Windows)
REM This batch file automates the setup process for Windows users

echo.
echo ========================================
echo  Ceramic Armor ML API - Quick Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.10 or 3.11 from python.org
    echo Then add Python to your PATH and try again
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo.
echo 🔧 Running automated setup...
echo.

REM Run the Python setup script
python setup.py

echo.
echo 📋 Quick Start Commands:
echo.
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Start the application:
echo    python start_server.py
echo.
echo 3. Open in browser:
echo    http://localhost:8000
echo.

pause