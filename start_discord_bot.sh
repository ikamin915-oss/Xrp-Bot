#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v screen >/dev/null 2>&1; then
  screen -dmS xrp-discord-bot python3 discord_bot.py
  echo "Started discord_bot.py in screen session: xrp-discord-bot"
else
  nohup python3 discord_bot.py >> discord_bot.out.log 2>&1 &
  echo "Started discord_bot.py with nohup. PID: $!"
fi
