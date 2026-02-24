@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

start "Backend (uvicorn)" cmd /k "uvicorn webapp.backend.app:app --host 127.0.0.1 --port 8000 --reload"

start "" "http://127.0.0.1:8000/"
