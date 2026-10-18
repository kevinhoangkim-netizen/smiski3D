#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local/npm-cache .local/matplotlib .local/cache
export MPLCONFIGDIR="$PWD/.local/matplotlib"
export XDG_CACHE_HOME="$PWD/.local/cache"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r backend/requirements.lock.txt
.venv/bin/python -m pip install --disable-pip-version-check -r backend/requirements-appearance.txt
if ! .venv/bin/python scripts/download_appearance_model.py; then
  printf '\nLocal outfit model download unavailable. Pose/manual controls remain functional.\n' >&2
fi
npm ci --prefix frontend --cache "$PWD/.local/npm-cache"
if ! .venv/bin/python scripts/download_model.py; then
  printf '\nPhoto model download unavailable. Manual mode is fully functional. See LIMITATIONS.md.\n' >&2
fi
npm run build --prefix frontend
.venv/bin/python -m pytest -q
