#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v screen >/dev/null 2>&1; then
  screen -S xrp-trade-bot -X quit >/dev/null 2>&1 || true
fi

pkill -f "$(pwd)/main.py" >/dev/null 2>&1 || true
pkill -f "python3 main.py" >/dev/null 2>&1 || true
echo "Stop signal sent for main.py trading bot."
