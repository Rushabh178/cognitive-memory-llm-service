@echo off
REM ============================================================
REM setup.bat — One-shot environment setup for Windows
REM
REM Run this once before starting the server for the first time:
REM   setup.bat
REM ============================================================

echo ----------------------------------------------------
echo  AI Cognitive Memory System — Environment Setup
echo ----------------------------------------------------

REM Step 1: Create a virtual environment in .\venv
REM A venv keeps all project dependencies isolated from your
REM system Python so different projects don't conflict.
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3.11+ is installed and on your PATH.
    pause
    exit /b 1
)

REM Step 2: Activate the virtual environment.
REM After this, python and pip refer to the venv copies.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 3: Install all dependencies from requirements.txt.
echo [3/4] Installing dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)

REM Step 4: Create .env from .env.example if .env doesn't exist yet.
REM We never overwrite an existing .env to avoid wiping real API keys.
echo [4/4] Checking .env file...
if not exist .env (
    copy .env.example .env >nul
    echo   --^> .env created from .env.example
    echo   --^> IMPORTANT: open .env and fill in ANTHROPIC_API_KEY and API_BEARER_TOKEN
) else (
    echo   --^> .env already exists, skipping copy
)

echo.
echo ----------------------------------------------------
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Fill in your API keys in .env
echo    2. The venv is already active in this window
echo    3. Start the server:   uvicorn main:app --reload
echo ----------------------------------------------------

REM Keep the window open so the user can read the output
pause
