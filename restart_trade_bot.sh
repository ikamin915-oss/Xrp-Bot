#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash ./stop_trade_bot.sh
sleep 2
bash ./start_trade_bot.sh
