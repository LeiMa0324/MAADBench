#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
AD_PYTHON="${PYTHON:-python3}"
if ! "$AD_PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "benchmark_AD requires the single project environment with Python 3.12." >&2
  echo "Activate .venv, or set PYTHON=/path/to/.venv/bin/python." >&2
  exit 2
fi
exec "$AD_PYTHON" -u -m benchmark_AD.run_all "$@"
