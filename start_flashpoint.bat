@echo off
REM FLASHPOINT V2 - Streaming State Machine - Windows Quick Start (FIXED)

echo ============================================================
echo ⚡ FLASHPOINT V2 - STREAMING STATE MACHINE
echo ============================================================
echo.
echo 🚀 Features:
echo   • Token-level streaming updates
echo   • Speculative execution (parallel branch evaluation)
echo   • Sub-10ms cycle latency target
echo   • Cascade failure injection
echo   • Partial JSON parsing
echo   • ✅ STRICT action filtering (no more alerts!)
echo.

echo 📋 Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Python is installed
    python --version
) else (
    echo ✗ Python not found! Please install Python 3.7+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo 📦 Step 2: Installing dependencies...
echo Running: pip install websockets openai
python -m pip install websockets openai

echo.
echo 🔑 Step 3: Setting up API key...
echo.
echo IMPORTANT: You need your Cerebras API key!
echo Get it from: https://cloud.cerebras.ai/
echo.
set /p CEREBRAS_API_KEY="Enter your Cerebras API key: "

if "%CEREBRAS_API_KEY%"=="" (
    echo ✗ No API key provided!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✓ SETUP COMPLETE!
echo ============================================================
echo.
echo 📝 NEXT STEPS:
echo.
echo 1. This window will now start the backend server
echo.
echo 2. Open flashpoint_ui.html in your browser
echo    (Just double-click the file)
echo.
echo 3. Watch for:
echo    • Streaming hypothesis tokens appearing in real-time
echo    • Latency comparator showing microseconds vs milliseconds
echo    • Severity dropping from 9 → 2 in 10 cycles
echo    • NO MORE ALERT ACTIONS - only real fixes!
echo.
echo ============================================================
echo.
echo Press any key to start the backend server...
pause >nul

echo.
echo 🚀 Starting FLASHPOINT V2 backend server...
echo.
echo 💡 TIP: Watch the terminal for cycle timing metrics!
echo 💡 Keep this window OPEN - it's running the server!
echo.

REM Run Python in the SAME session so environment variable persists
python flashpoint_engine.py

echo.
echo.
echo ============================================================
echo Server stopped. Press any key to exit...
pause >nul
