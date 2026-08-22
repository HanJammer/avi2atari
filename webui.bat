@echo off
rem One-click local AVF Suite web UI: venv + deps + launch (YouTube ON locally).
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\python -m pip install -q --upgrade pip
call .venv\Scripts\python -m pip install -q -r web\requirements.txt
echo Open http://127.0.0.1:8000  (Ctrl+C to stop)
.venv\Scripts\python web\app.py
