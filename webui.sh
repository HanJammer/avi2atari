#!/bin/sh
# One-click local AVF Suite web UI: venv + deps + launch (YouTube ON locally).
cd "$(dirname "$0")" || exit 1
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
[ -d .venv ] || "$PY" -m venv .venv
./.venv/bin/pip install -q --upgrade pip >/dev/null 2>&1
./.venv/bin/pip install -q -r requirements.txt || exit 1
echo "Open http://127.0.0.1:8000  (Ctrl+C to stop)"
exec ./.venv/bin/python web/app.py
