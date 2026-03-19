@echo off
setlocal
cd /d "%~dp0"

if not exist ".venvstickman\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found: .venvstickman
    echo Create it first with:
    echo     py -3.12 -m venv .venvstickman
    pause
    exit /b 1
)

call ".venvstickman\Scripts\activate.bat"

if not exist "main.py" (
    echo [ERROR] main.py not found in project root.
    pause
    exit /b 1
)

python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] The project exited with an error.
    pause
)
