#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p data/intake_full checkpoints logs outputs
printf '\nHannah environment ready.\n'
printf 'Activate with: source .venv/bin/activate\n'
printf 'Run tests with: ./scripts/run_tests.sh\n'
printf 'Train with: python scripts/train_historical.py --data-dir data/intake_full\n'
