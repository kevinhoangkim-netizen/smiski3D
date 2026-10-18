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
  printf '\nLocal outfit model download unavailable. Photo appearance analysis will report the missing model.\n' >&2
fi
npm ci --prefix frontend --cache "$PWD/.local/npm-cache"
if ! .venv/bin/python scripts/download_model.py; then
  printf '\nPhoto model download unavailable. Photo generation needs this model. See LIMITATIONS.md.\n' >&2
fi
.venv/bin/python scripts/download_people_model.py
.venv/bin/python scripts/download_garment_model.py
npm run build --prefix frontend
.venv/bin/python -m pytest -q
