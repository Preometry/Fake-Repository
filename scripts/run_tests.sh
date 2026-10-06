#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
export PYTHONPATH="$ROOT/src/hannah_model${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" tests/test_trust_hannah.py
"$PYTHON" scripts/independent_geodesic_checks.py
