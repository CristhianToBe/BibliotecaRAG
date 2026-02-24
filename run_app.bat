@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

start "Backend (uvicorn)" cmd /k "uvicorn webapp.backend.app:app --host 127.0.0.1 --port 8000 --reload"
start "Frontend (http.server)" cmd /k "python -m http.server 5173 --bind 127.0.0.1 --directory webapp/frontend"

start "" "http://127.0.0.1:5173/index.html"
