#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v screen >/dev/null 2>&1; then
  screen -dmS xrp-trade-bot python3 main.py
  echo "Started main.py in screen session: xrp-trade-bot"
else
  nohup python3 main.py >> trade_bot.out.log 2>&1 &
  echo "Started main.py with nohup. PID: $!"
fi
