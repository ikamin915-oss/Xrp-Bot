#!/usr/bin/env python3
"""Binance Futures bot with safe dry-run defaults.

Usage:
    1. Install dependencies: pip install -r requirements.txt
    2. Update .env with your Binance Futures testnet or live credentials
    3. Run: python main.py

The bot calculates EMA7, EMA25, and EMA99 on closed 5-minute candles, while
entries require strict EMA7/EMA25 pullback confirmation. It supports:
    - Public market-data reads without API keys
    - Dry-run mode by default
    - Futures leverage configuration for live trading
    - Position-aware entries, exits, daily safety limits, and Discord alerts

This example assumes one-way mode on Binance Futures.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import uuid4

import pandas as pd
import requests
from binance.error import ClientError, ServerError
from binance.um_futures import UMFutures
from dotenv import load_dotenv

try:
    import certifi
except ImportError:  # pragma: no cover - requirements.txt includes certifi.
    certifi = None


FUTURES_LIVE_URL = "https://fapi.binance.me"
FUTURES_TESTNET_URL = "https://demo-fapi.binance.com"
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
BOT_LOG_PATH = PROJECT_DIR / "bot.log"
TRADES_CSV_PATH = PROJECT_DIR / "trades.csv"
TRADE_ANALYZER_PATH = PROJECT_DIR / "trade_analyzer.py"
LEARNING_REPORT_PATH = PROJECT_DIR / "learning_report.md"
LEARNING_STATE_PATH = PROJECT_DIR / "learning_state.json"
FUTURES_TESTNET_KEYS_ERROR = (
    "Your keys are not valid for Binance Futures Demo/Testnet. "
    "Create new USD-M Futures Demo API keys and put them in .env."
)
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]
TRADE_LOG_FIELDS = [
    "trade_id",
    "timestamp",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl_usdt",
    "exit_reason",
    "entry_reason",
    "ema7",
    "ema25",
    "ema99",
    "ema_spread_pct",
    "candle_body_ratio",
    "distance_from_ema7_pct",
    "volume",
    "previous_candle_direction",
    "cooldown_status",
    "tp1_hit_before_exit",
    "holding_time_seconds",
]
SUPPORTED_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}
DISCORD_TIMEOUT_SECONDS = 10
DISCORD_MAX_FIELD_VALUE_LENGTH = 1024
DISCORD_MAX_DESCRIPTION_LENGTH = 4096
DISCORD_MAX_EMBED_FIELDS = 25
DISCORD_EMBED_COLORS = {
    "success": 0x2ECC71,
    "info": 0x3498DB,
    "warning": 0xF1C40F,
    "danger": 0xE74C3C,
    "learning": 0x1ABC9C,
    "neutral": 0x95A5A6,
}


class ConfigError(ValueError):
    """Raised when required configuration is invalid."""


@dataclass(frozen=True)
class BotConfig:
    api_key: Optional[str]
    api_secret: Optional[str]
    symbol: str
    interval: str
    candle_limit: int
    ema_fast: int
    ema_slow: int
    ema_trend: int
    order_size_quote: Decimal
    leverage: int
    stop_loss_pct: Decimal
    tp1_profit_pct: Decimal
    tp2_profit_pct: Decimal
    tp1_close_ratio: Decimal
    use_testnet: bool
    dry_run: bool
    live_trading: bool
    confirm_live: bool
    run_once: bool
    loop_interval_seconds: int
    request_timeout: int
    log_level: str
    discord_webhook_url: Optional[str]
    heartbeat_interval_minutes: int
    max_trades_per_day: int
    max_daily_loss_usdt: Decimal
    max_daily_profit_usdt: Decimal
    stop_after_losses: int
    analyze_on_start: bool
    auto_apply_learning: bool
    min_trades_before_learning: int

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def mode_label(self) -> str:
        if self.dry_run:
            return "DRY_RUN"
        if self.can_place_real_orders:
            return "LIVE"
        return "LIVE_BLOCKED"

    @property
    def can_place_real_orders(self) -> bool:
        return (
            not self.dry_run
            and self.confirm_live
            and self.has_credentials
        )

    @property
    def order_block_reason(self) -> Optional[str]:
        if self.dry_run:
            return "DRY_RUN=true"
        if not self.confirm_live:
            return "CONFIRM_LIVE is not true"
        if not self.has_credentials:
            return "API credentials are missing"
        return None

    @classmethod
    def from_env(cls) -> "BotConfig":
        load_dotenv(dotenv_path=ENV_PATH, override=True)

        symbol = get_env("SYMBOL", "XRPUSDT").upper()
        interval = get_env("INTERVAL", "5m")

        config = cls(
            api_key=normalize_secret(os.getenv("BINANCE_API_KEY")),
            api_secret=normalize_secret(os.getenv("BINANCE_API_SECRET")),
            symbol=symbol,
            interval=interval,
            candle_limit=get_int_env("CANDLE_LIMIT", 150),
            ema_fast=get_int_env("EMA_FAST", 7),
            ema_slow=get_int_env("EMA_SLOW", 25),
            ema_trend=get_int_env("EMA_TREND", 99),
            order_size_quote=get_order_size_quote_env("6"),
            leverage=get_int_env("LEVERAGE", 5),
            stop_loss_pct=get_decimal_env("STOP_LOSS_PCT", "0.35"),
            tp1_profit_pct=get_decimal_env("TP1_PROFIT_PCT", "0.5"),
            tp2_profit_pct=get_decimal_env("TP2_PROFIT_PCT", "0.9"),
            tp1_close_ratio=get_decimal_env("TP1_CLOSE_RATIO", "0.6"),
            use_testnet=get_bool_env("USE_TESTNET", True),
            dry_run=get_bool_env("DRY_RUN", True),
            live_trading=get_bool_env("LIVE_TRADING", False),
            confirm_live=get_bool_env("CONFIRM_LIVE", False),
            run_once=get_bool_env("RUN_ONCE", False),
            loop_interval_seconds=get_int_env("LOOP_INTERVAL_SECONDS", 60),
            request_timeout=get_int_env("REQUEST_TIMEOUT", 20),
            log_level=get_env("LOG_LEVEL", "INFO").upper(),
            discord_webhook_url=normalize_optional_value(os.getenv("DISCORD_WEBHOOK_URL")),
            heartbeat_interval_minutes=get_int_env("HEARTBEAT_INTERVAL_MINUTES", 15),
            max_trades_per_day=get_int_env("MAX_TRADES_PER_DAY", 0),
            max_daily_loss_usdt=get_decimal_env("MAX_DAILY_LOSS_USDT", "0.50"),
            max_daily_profit_usdt=get_decimal_env("MAX_DAILY_PROFIT_USDT", "1.00"),
            stop_after_losses=get_int_env("STOP_AFTER_LOSSES", 2),
            analyze_on_start=get_bool_env("ANALYZE_ON_START", False),
            auto_apply_learning=get_bool_env("AUTO_APPLY_LEARNING", False),
            min_trades_before_learning=get_int_env("MIN_TRADES_BEFORE_LEARNING", 20),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            raise ConfigError(
                f"Unsupported INTERVAL '{self.interval}'. Use one of: {', '.join(sorted(SUPPORTED_INTERVALS))}"
            )
        if self.interval != "5m":
            raise ConfigError("This pullback/rejection strategy is locked to INTERVAL=5m.")
        if self.ema_fast <= 0 or self.ema_slow <= 0 or self.ema_trend <= 0:
            raise ConfigError("EMA_FAST, EMA_SLOW, and EMA_TREND must be positive integers.")
        if not self.ema_fast < self.ema_slow < self.ema_trend:
            raise ConfigError("EMA settings must satisfy EMA_FAST < EMA_SLOW < EMA_TREND.")
        if self.candle_limit < self.ema_trend + 5:
            raise ConfigError("CANDLE_LIMIT is too small for the selected indicators.")
        if not Decimal("0") < self.order_size_quote:
            raise ConfigError("ORDER_SIZE_USDC/ORDER_SIZE_USDT must be greater than 0.")
        if self.leverage <= 0:
            raise ConfigError("LEVERAGE must be greater than 0.")
        if not Decimal("0") < self.stop_loss_pct < Decimal("100"):
            raise ConfigError("STOP_LOSS_PCT must be between 0 and 100.")
        if not Decimal("0") < self.tp1_profit_pct < self.tp2_profit_pct < Decimal("100"):
            raise ConfigError("TP settings must satisfy 0 < TP1_PROFIT_PCT < TP2_PROFIT_PCT < 100.")
        if not Decimal("0") < self.tp1_close_ratio < Decimal("1"):
            raise ConfigError("TP1_CLOSE_RATIO must be between 0 and 1.")
        if self.loop_interval_seconds <= 0:
            raise ConfigError("LOOP_INTERVAL_SECONDS must be greater than 0.")
        if self.request_timeout <= 0:
            raise ConfigError("REQUEST_TIMEOUT must be greater than 0.")
        if self.heartbeat_interval_minutes <= 0:
            raise ConfigError("HEARTBEAT_INTERVAL_MINUTES must be greater than 0.")
        if self.max_trades_per_day < 0:
            raise ConfigError("MAX_TRADES_PER_DAY must be 0 for unlimited or greater than 0 for a daily cap.")
        if not Decimal("0") < self.max_daily_loss_usdt:
            raise ConfigError("MAX_DAILY_LOSS_USDT must be greater than 0.")
        if not Decimal("0") < self.max_daily_profit_usdt:
            raise ConfigError("MAX_DAILY_PROFIT_USDT must be greater than 0.")
        if self.stop_after_losses <= 0:
            raise ConfigError("STOP_AFTER_LOSSES must be greater than 0.")
        if self.min_trades_before_learning <= 0:
            raise ConfigError("MIN_TRADES_BEFORE_LEARNING must be greater than 0.")
        if not self.dry_run and self.confirm_live and not self.has_credentials:
            raise ConfigError(
                "Non-dry-run trading requires BINANCE_API_KEY and BINANCE_API_SECRET in .env."
            )


@dataclass(frozen=True)
class SymbolRules:
    min_qty: Decimal
    max_qty: Decimal
    qty_step: Decimal
    tick_size: Decimal
    min_notional: Decimal


@dataclass(frozen=True)
class Position:
    quantity: Decimal
    entry_price: Decimal

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "LONG"
        if self.quantity < 0:
            return "SHORT"
        return "FLAT"


@dataclass(frozen=True)
class TradingSignal:
    action: str
    reason: str
    close_price: Decimal
    ema_fast: Decimal
    ema_slow: Decimal
    ema_trend: Decimal
    ema_spread_pct: Decimal
    candle_body_ratio: Decimal
    distance_from_ema7_pct: Decimal
    volume: Decimal
    previous_candle_direction: str
    candle_close_time: pd.Timestamp


@dataclass
class PositionExitState:
    key: str
    original_quantity: Decimal
    tp1_done: bool = False
    break_even_armed: bool = False


@dataclass
class TradeContext:
    trade_id: str
    key: str
    opened_at: datetime
    symbol: str
    side: str
    entry_price: Decimal
    entry_reason: str
    ema_fast: Decimal
    ema_slow: Decimal
    ema_trend: Decimal
    ema_spread_pct: Decimal
    candle_body_ratio: Decimal
    distance_from_ema7_pct: Decimal
    volume: Decimal
    previous_candle_direction: str
    cooldown_status: str
    tp1_hit_before_exit: bool = False


class BinanceFuturesBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.futures_base_url = FUTURES_TESTNET_URL if config.use_testnet else FUTURES_LIVE_URL
        client_kwargs: dict[str, Any] = {
            "key": config.api_key,
            "secret": config.api_secret,
            "timeout": config.request_timeout,
        }
        client_kwargs["base_url"] = self.futures_base_url
        self.client = UMFutures(**client_kwargs)
        self.symbol_rules: Optional[SymbolRules] = None
        self.current_day: date = utc_today()
        self.trades_today = 0
        self.losses_today = 0
        self.daily_loss_usdt = Decimal("0")
        self.daily_profit_usdt = Decimal("0")
        self.last_known_position = Position(quantity=Decimal("0"), entry_price=Decimal("0"))
        self.last_signal = "NONE"
        self.last_heartbeat_at = time.monotonic()
        self.last_signal_alert_key: Optional[str] = None
        self.trade_shutdown_reason: Optional[str] = None
        self.trade_shutdown_notified = False
        self.exit_state: Optional[PositionExitState] = None
        self.current_trade_context: Optional[TradeContext] = None
        self.learning_analysis_triggers_run: set[str] = set()

    def run(self) -> None:
        self.logger.info("Loaded environment file: %s", ENV_PATH)
        self.logger.info(
            "Startup flags: DRY_RUN=%s USE_TESTNET=%s CONFIRM_LIVE=%s computed mode=%s",
            self.config.dry_run,
            self.config.use_testnet,
            self.config.confirm_live,
            self.config.mode_label,
        )
        self.logger.info(
            "Risk settings: LEVERAGE=%sx requested_order_size=%s",
            self.config.leverage,
            format_decimal(self.config.order_size_quote),
        )
        self.logger.info(
            "Learning settings: ANALYZE_ON_START=%s AUTO_APPLY_LEARNING=%s MIN_TRADES_BEFORE_LEARNING=%s",
            self.config.analyze_on_start,
            self.config.auto_apply_learning,
            self.config.min_trades_before_learning,
        )
        self.warn_learning_auto_apply_if_needed()
        if self.config.analyze_on_start:
            self.run_learning_analysis("startup")
        self.logger.info("USE_TESTNET=%s", self.config.use_testnet)
        self.logger.info(
            "Selected Binance Futures API base URL: %s",
            self.futures_base_url,
        )
        self.logger.info("symbol=%s", self.config.symbol)
        self.logger.info("API key exists=%s", bool(self.config.api_key))
        self.warn_live_mode_if_needed()
        self.verify_connectivity()
        self.validate_futures_public_endpoints()
        self.symbol_rules = self.fetch_symbol_rules()
        self.logger.info(
            "Bot ready for %s on %s (%s mode)",
            self.config.symbol,
            "testnet" if self.config.use_testnet else "live futures",
            self.config.mode_label,
        )

        if self.config.can_place_real_orders:
            self.configure_leverage()

        self.log_startup_url(
            "startup mark_price",
            "GET",
            self.futures_api_url("/fapi/v1/premiumIndex", {"symbol": self.config.symbol}),
        )
        startup_price = self.get_mark_price()
        if self.config.has_credentials:
            self.log_startup_url(
                "startup position_risk",
                "GET",
                self.futures_api_url("/fapi/v2/positionRisk", {"symbol": self.config.symbol}),
            )
        startup_position = self.get_position_snapshot()
        send_discord_message(
            self.build_status_message(
                event="Bot started",
                status="RUNNING",
                mark_price=startup_price,
                position=startup_position,
            )
        )
        self.last_heartbeat_at = time.monotonic()

        while True:
            try:
                self.run_cycle()
            except Exception as exc:
                self.logger.exception("Bot cycle failed. Software-managed position monitoring is interrupted.")
                send_discord_message(
                    f"Bot cycle error\n"
                    f"symbol: {self.config.symbol}\n"
                    f"error: {exc}\n"
                    f"action: defensive close if a live position is open"
                )
                self.close_open_position_defensively(
                    f"Software-managed monitoring failed: {exc}"
                )
                raise
            if self.config.run_once:
                return
            self.logger.info(
                "Sleeping for %s seconds before the next cycle.",
                self.config.loop_interval_seconds,
            )
            time.sleep(self.config.loop_interval_seconds)

    def warn_live_mode_if_needed(self) -> None:
        if self.config.dry_run:
            return

        if not self.config.confirm_live:
            warning = "LIVE BLOCKED - missing CONFIRM_LIVE=true"
            self.logger.warning(
                "%s",
                warning,
            )
            print(warning, file=sys.stderr)
            return

        if not self.config.has_credentials:
            warning = "LIVE BLOCKED - missing API credentials"
            self.logger.warning("%s", warning)
            print(warning, file=sys.stderr)
            return

        if not self.config.use_testnet:
            warning = "LIVE TRADING ENABLED"
            self.logger.warning(warning)
            print(warning, file=sys.stderr)
        else:
            self.logger.warning("TESTNET TRADING ENABLED")

    def warn_learning_auto_apply_if_needed(self) -> None:
        if not self.config.auto_apply_learning:
            return
        warning = "Auto-apply learning is disabled for safety."
        self.logger.warning(warning)
        print(warning, file=sys.stderr)

    def futures_api_url(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.futures_base_url.rstrip('/')}{normalized_path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        self.validate_futures_url(url)
        return url

    def validate_futures_url(self, url: str) -> None:
        allowed_bases = (FUTURES_LIVE_URL, FUTURES_TESTNET_URL)
        if not url.startswith(allowed_bases):
            raise ConfigError(f"Blocked non-futures Binance URL: {url}")

        lowered = url.lower()
        live_domain = FUTURES_LIVE_URL.removeprefix("https://fapi.")
        website_host = f"www.{live_domain}"
        website_path = f"{live_domain}/" + "en"
        if website_host in lowered or website_path in lowered:
            raise ConfigError(f"Blocked Binance website URL: {url}")

    def log_startup_url(self, label: str, method: str, url: str) -> None:
        self.validate_futures_url(url)
        self.logger.info("Startup URL [%s]: %s %s", label, method.upper(), url)

    def verify_connectivity(self) -> None:
        url = self.futures_api_url("/fapi/v1/time")
        self.log_startup_url("connectivity time", "GET", url)
        response = requests.get(
            url,
            timeout=self.config.request_timeout,
            allow_redirects=False,
            verify=certifi.where() if certifi else True,
        )
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location", "")
            raise ConfigError(
                "Binance Futures connectivity check attempted to redirect. "
                f"Only Futures API URLs are allowed. url={url} location={location}"
            )
        response.raise_for_status()
        server_time = response.json().get("serverTime")
        self.logger.info("Binance Futures API reachable. serverTime=%s", server_time)

    def validate_futures_public_endpoints(self) -> None:
        try:
            self.log_startup_url(
                "public exchange_info validation",
                "GET",
                self.futures_api_url("/fapi/v1/exchangeInfo"),
            )
            self.client.exchange_info()
            self.logger.info("Startup validation passed: client.exchange_info()")
        except ClientError as exc:
            error_details = format_binance_api_error(exc)
            self.logger.warning("Startup public exchange_info check failed: %s", error_details)
            return
        except (ServerError, requests.RequestException) as exc:
            self.logger.warning("Startup public exchange_info request failed: %s", exc)
            return

        try:
            self.log_startup_url(
                "public mark_price validation",
                "GET",
                self.futures_api_url("/fapi/v1/premiumIndex", {"symbol": self.config.symbol}),
            )
            self.client.mark_price(symbol=self.config.symbol)
            self.logger.info("Startup validation passed: client.mark_price(symbol=%s)", self.config.symbol)
        except ClientError as exc:
            self.logger.warning("Optional startup mark_price check failed: %s", format_binance_api_error(exc))
        except (ServerError, requests.RequestException) as exc:
            self.logger.warning("Optional startup mark_price request failed: %s", exc)

    def fetch_symbol_rules(self) -> SymbolRules:
        self.log_startup_url(
            "symbol rules exchange_info",
            "GET",
            self.futures_api_url("/fapi/v1/exchangeInfo"),
        )
        exchange_info = self.client.exchange_info()
        symbol_info = next(
            (item for item in exchange_info["symbols"] if item["symbol"] == self.config.symbol),
            None,
        )
        if not symbol_info:
            raise ConfigError(f"Symbol {self.config.symbol} was not found on Binance Futures.")

        filters = {item["filterType"]: item for item in symbol_info.get("filters", [])}
        qty_filter = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        price_filter = filters.get("PRICE_FILTER")
        min_notional_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL")

        if not qty_filter or not price_filter:
            raise ConfigError(f"Could not load trading rules for {self.config.symbol}.")

        return SymbolRules(
            min_qty=Decimal(qty_filter["minQty"]),
            max_qty=Decimal(qty_filter["maxQty"]),
            qty_step=Decimal(qty_filter["stepSize"]),
            tick_size=Decimal(price_filter["tickSize"]),
            min_notional=Decimal(
                (min_notional_filter or {}).get("notional")
                or (min_notional_filter or {}).get("minNotional", "0")
            ),
        )

    def configure_leverage(self) -> None:
        self.logger.info(
            "Setting leverage for %s to %sx.",
            self.config.symbol,
            self.config.leverage,
        )
        self.log_startup_url(
            "change leverage",
            "POST",
            self.futures_api_url("/fapi/v1/leverage", {"symbol": self.config.symbol}),
        )
        self.client.change_leverage(
            symbol=self.config.symbol,
            leverage=self.config.leverage,
        )

    def run_cycle(self) -> None:
        assert self.symbol_rules is not None

        self.reset_daily_counters_if_needed()
        candles = self.fetch_candles()
        signal = self.generate_signal(candles)
        mark_price = self.get_mark_price()
        position = self.get_position() if self.config.has_credentials else Position(
            quantity=Decimal("0"),
            entry_price=Decimal("0"),
        )
        self.last_known_position = position
        self.last_signal = f"{signal.action} | {signal.reason}"

        self.logger.info(
            "Signal=%s | close=%s | mark=%s | EMA(%s)=%s | EMA(%s)=%s | EMA(%s)=%s | reason=%s",
            signal.action,
            signal.close_price,
            mark_price,
            self.config.ema_fast,
            signal.ema_fast,
            self.config.ema_slow,
            signal.ema_slow,
            self.config.ema_trend,
            signal.ema_trend,
            signal.reason,
        )
        self.logger.info(
            "Current position: side=%s quantity=%s entry=%s",
            position.side,
            format_decimal(abs(position.quantity)),
            format_decimal(position.entry_price),
        )

        if signal.action != "HOLD":
            self.maybe_send_signal_alert(signal, mark_price)
        self.maybe_send_heartbeat(mark_price, position)

        if self.config.can_place_real_orders:
            if position.is_open:
                self.ensure_software_monitoring_state(position)
            else:
                self.exit_state = None
                self.cancel_all_reduce_only_orders()

        if position.is_open:
            if self.handle_exit_rules(position, mark_price):
                self.logger.info("Exit action handled. Waiting for the next cycle.")
                return

        if signal.action == "HOLD":
            self.logger.info("No entry signal this cycle.")
            return

        if position.is_open:
            if signal.action == position.side:
                self.logger.info("Position already exists on the signal side. No new order sent.")
                return
            self.logger.info(
                "Opposite signal detected against the open %s position. Closing only and waiting until next cycle.",
                position.side,
            )
            if self.config.dry_run:
                self.logger.info(
                    "Dry-run enabled. Planned close only for the open %s position.",
                    position.side,
                )
                return
            if not self.config.can_place_real_orders:
                self.log_order_block("closing opposite position")
                return
            self.close_position(position, mark_price, f"Opposite {signal.action} signal detected.")
            return

        trade_block_reason = self.peek_trade_block_reason()
        if trade_block_reason:
            self.handle_trade_block(trade_block_reason, signal.action, mark_price, position)
            return

        planned_quantity = self.calculate_order_quantity(mark_price)

        if self.config.dry_run:
            self.log_dry_run_entry_plan(signal.action, planned_quantity, mark_price)
            return
        if not self.config.can_place_real_orders:
            self.log_order_block(f"opening {signal.action} position")
            self.log_dry_run_entry_plan(signal.action, planned_quantity, mark_price)
            return

        self.open_position(signal, planned_quantity, mark_price)

    def fetch_candles(self) -> pd.DataFrame:
        raw_klines = self.client.klines(
            symbol=self.config.symbol,
            interval=self.config.interval,
            limit=self.config.candle_limit,
        )
        if len(raw_klines) < self.config.ema_trend + 3:
            raise RuntimeError("Not enough candles returned from Binance to build indicators.")

        frame = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = frame[column].astype(float)

        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        frame["ema_fast"] = frame["close"].ewm(span=self.config.ema_fast, adjust=False).mean()
        frame["ema_slow"] = frame["close"].ewm(span=self.config.ema_slow, adjust=False).mean()
        frame["ema_trend"] = frame["close"].ewm(span=self.config.ema_trend, adjust=False).mean()
        return frame

    def generate_signal(self, candles: pd.DataFrame) -> TradingSignal:
        if len(candles) < 3:
            raise RuntimeError("At least three candles are required to evaluate closed-candle signals.")

        closed = candles.iloc[:-1].copy()
        if len(closed) < 2:
            raise RuntimeError("Not enough closed candles returned to generate a signal.")

        latest = closed.iloc[-1]
        previous = closed.iloc[-2]

        bullish_latest = latest["close"] > latest["open"]
        bearish_latest = latest["close"] < latest["open"]
        long_previous_touch = self.previous_candle_touched_ema(previous, side="LONG")
        short_previous_touch = self.previous_candle_touched_ema(previous, side="SHORT")

        long_setup = all(
            (
                latest["close"] > latest["ema_slow"],
                latest["ema_fast"] > latest["ema_slow"],
                long_previous_touch,
                bullish_latest,
                latest["close"] > previous["close"],
            )
        )
        short_setup = all(
            (
                latest["close"] < latest["ema_slow"],
                latest["ema_fast"] < latest["ema_slow"],
                short_previous_touch,
                bearish_latest,
                latest["close"] < previous["close"],
            )
        )

        action = "HOLD"
        if long_setup:
            reason = "Strict bullish confirmation: previous candle touched EMA7/EMA25 and current candle closed bullish above previous close."
            action = "LONG"
        elif short_setup:
            reason = "Strict bearish confirmation: previous candle touched EMA7/EMA25 and current candle closed bearish below previous close."
            action = "SHORT"
        else:
            reason = self.build_hold_reason(
                latest=latest,
                previous=previous,
                bullish_latest=bullish_latest,
                bearish_latest=bearish_latest,
                long_previous_touch=long_previous_touch,
                short_previous_touch=short_previous_touch,
            )

        close_price = decimal_from_number(latest["close"])
        ema_fast = decimal_from_number(latest["ema_fast"])
        ema_slow = decimal_from_number(latest["ema_slow"])
        ema_trend = decimal_from_number(latest["ema_trend"])
        candle_range = decimal_from_number(latest["high"]) - decimal_from_number(latest["low"])
        candle_body = abs(decimal_from_number(latest["close"]) - decimal_from_number(latest["open"]))
        candle_body_ratio = safe_decimal_ratio(candle_body, candle_range)
        ema_spread_pct = safe_decimal_ratio(ema_fast - ema_slow, close_price) * Decimal("100")
        distance_from_ema7_pct = safe_decimal_ratio(abs(close_price - ema_fast), close_price) * Decimal("100")

        return TradingSignal(
            action=action,
            reason=reason,
            close_price=close_price,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_trend=ema_trend,
            ema_spread_pct=ema_spread_pct,
            candle_body_ratio=candle_body_ratio,
            distance_from_ema7_pct=distance_from_ema7_pct,
            volume=decimal_from_number(latest["volume"]),
            previous_candle_direction=candle_direction(previous),
            candle_close_time=latest["close_time"],
        )

    def get_mark_price(self) -> Decimal:
        price_data = self.client.mark_price(symbol=self.config.symbol)
        return Decimal(price_data["markPrice"])

    def get_position_snapshot(self) -> Position:
        if not self.config.has_credentials:
            return Position(quantity=Decimal("0"), entry_price=Decimal("0"))
        try:
            return self.get_position()
        except (ClientError, ServerError, requests.RequestException) as exc:
            self.logger.warning("Could not fetch startup position snapshot: %s", exc)
            return Position(quantity=Decimal("0"), entry_price=Decimal("0"))

    def get_position(self) -> Position:
        positions = self.client.get_position_risk(symbol=self.config.symbol)
        for item in positions:
            quantity = Decimal(item["positionAmt"])
            if quantity != 0:
                entry_price = Decimal(item["entryPrice"])
                return Position(quantity=quantity, entry_price=entry_price)
        return Position(quantity=Decimal("0"), entry_price=Decimal("0"))

    def handle_exit_rules(self, position: Position, mark_price: Decimal) -> bool:
        self.sync_exit_state(position)
        if self.exit_state is None:
            return False

        if self.config.can_place_real_orders:
            return self.handle_software_take_profit_rules(position, mark_price)

        return self.handle_dry_run_exit_rules(position, mark_price)

    def handle_dry_run_exit_rules(self, position: Position, mark_price: Decimal) -> bool:
        assert self.exit_state is not None

        profit_pct = calculate_unrealized_profit_pct(position, mark_price)

        if not self.exit_state.tp1_done:
            stop_reason = self.initial_stop_reason(position, mark_price)
            if stop_reason:
                return self.handle_full_exit(position, mark_price, stop_reason)

            if profit_pct >= self.config.tp1_profit_pct:
                quantity = self.calculate_tp1_close_quantity(position)
                reason = (
                    f"TP1 hit at {format_decimal(profit_pct)}% profit. "
                    f"Closing {format_decimal(self.config.tp1_close_ratio * Decimal('100'))}% and moving SL to break-even."
                )
                return self.handle_tp1_exit(
                    position=position,
                    mark_price=mark_price,
                    quantity=quantity,
                    reason=reason,
                    tp2_already_reached=profit_pct >= self.config.tp2_profit_pct,
                )

            return False

        if self.tp2_reached(position, mark_price):
            reason = f"TP2 hit at {format_decimal(profit_pct)}% profit. Closing remaining position."
            return self.handle_full_exit(position, mark_price, reason)

        break_even_reason = self.break_even_stop_reason(position, mark_price)
        if break_even_reason:
            return self.handle_full_exit(position, mark_price, break_even_reason)

        return False

    def handle_software_take_profit_rules(self, position: Position, mark_price: Decimal) -> bool:
        assert self.exit_state is not None

        profit_pct = calculate_unrealized_profit_pct(position, mark_price)

        if not self.exit_state.tp1_done:
            stop_reason = self.initial_stop_reason(position, mark_price)
            if stop_reason:
                return self.handle_full_exit(position, mark_price, stop_reason)

            if profit_pct >= self.config.tp1_profit_pct:
                quantity = self.calculate_tp1_close_quantity(position)
                reason = (
                    f"TP1 hit at {format_decimal(profit_pct)}% profit. "
                    f"Closing {format_decimal(self.config.tp1_close_ratio * Decimal('100'))}% and moving SL to break-even."
                )
                return self.handle_tp1_exit(
                    position=position,
                    mark_price=mark_price,
                    quantity=quantity,
                    reason=reason,
                    tp2_already_reached=profit_pct >= self.config.tp2_profit_pct,
                )

            return False

        if self.tp2_reached(position, mark_price):
            reason = f"TP2 hit at {format_decimal(profit_pct)}% profit. Closing remaining position."
            return self.handle_full_exit(position, mark_price, reason)

        break_even_reason = self.break_even_stop_reason(position, mark_price)
        if break_even_reason:
            return self.handle_full_exit(position, mark_price, break_even_reason)

        return False

    def sync_exit_state(self, position: Position) -> None:
        if not position.is_open:
            self.exit_state = None
            return

        position_key = build_position_key(position)
        if self.exit_state is None or self.exit_state.key != position_key:
            self.exit_state = PositionExitState(
                key=position_key,
                original_quantity=abs(position.quantity),
            )

    def ensure_software_monitoring_state(self, position: Position) -> None:
        if not position.is_open:
            self.exit_state = None
            return

        position_key = build_position_key(position)
        if self.exit_state is None or self.exit_state.key != position_key:
            self.install_software_managed_exit_state(position)
        if self.current_trade_context is None or self.current_trade_context.key != position_key:
            self.current_trade_context = self.build_recovered_trade_context(position)

    def install_software_managed_exit_state(self, position: Position) -> PositionExitState:
        state = PositionExitState(
            key=build_position_key(position),
            original_quantity=abs(position.quantity),
        )
        tp1_quantity = self.calculate_tp1_quantity_from_original(state.original_quantity)
        tp2_quantity = round_to_step(
            state.original_quantity - tp1_quantity,
            self.symbol_rules.qty_step,
        )
        self.validate_close_quantity(tp1_quantity, "TP1")
        self.validate_close_quantity(tp2_quantity, "TP2")

        self.exit_state = state

        warning = "Using software-managed stops. Keep bot running. Exchange-side stop disabled."
        stop_price = self.calculate_stop_price(position, break_even=False)
        tp1_price = self.calculate_take_profit_price(position, self.config.tp1_profit_pct)
        tp2_price = self.calculate_take_profit_price(position, self.config.tp2_profit_pct)
        self.logger.warning(warning)
        self.logger.info(
            "Software-managed levels: side=%s entry=%s stop_loss=%s tp1=%s tp2=%s "
            "position_size=%s tp1_quantity=%s tp2_quantity=%s",
            position.side,
            format_decimal(position.entry_price),
            format_decimal(stop_price),
            format_decimal(tp1_price),
            format_decimal(tp2_price),
            format_decimal(abs(position.quantity)),
            format_decimal(tp1_quantity),
            format_decimal(tp2_quantity),
        )
        send_discord_message(
            f"{warning}\n"
            f"symbol: {self.config.symbol}\n"
            f"side: {position.side}\n"
            f"entry: {format_decimal(position.entry_price)}\n"
            f"stop_loss: {format_decimal(stop_price)}\n"
            f"TP1: {format_decimal(tp1_price)}\n"
            f"TP2: {format_decimal(tp2_price)}\n"
            f"position_size: {format_decimal(abs(position.quantity))}"
        )
        return state

    def arm_software_break_even(self, position: Position) -> None:
        assert self.exit_state is not None
        self.exit_state.break_even_armed = True

        message = (
            f"Software break-even stop armed\n"
            f"symbol: {self.config.symbol}\n"
            f"side: {position.side}\n"
            f"entry: {format_decimal(position.entry_price)}\n"
            f"remaining_quantity: {format_decimal(abs(position.quantity))}\n"
            f"action: close with reduceOnly MARKET if mark price returns to entry"
        )
        self.logger.info(message.replace("\n", " | "))
        send_discord_message(message)

    def calculate_stop_price(self, position: Position, break_even: bool) -> Decimal:
        if break_even:
            return self.round_price_to_tick(position.entry_price)

        stop_loss_ratio = self.config.stop_loss_pct / Decimal("100")
        if position.side == "LONG":
            return self.round_price_to_tick(position.entry_price * (Decimal("1") - stop_loss_ratio))
        if position.side == "SHORT":
            return self.round_price_to_tick(position.entry_price * (Decimal("1") + stop_loss_ratio))
        raise ConfigError("Cannot calculate stop price for a flat position.")

    def calculate_take_profit_price(self, position: Position, profit_pct: Decimal) -> Decimal:
        profit_ratio = profit_pct / Decimal("100")
        if position.side == "LONG":
            return self.round_price_to_tick(position.entry_price * (Decimal("1") + profit_ratio))
        if position.side == "SHORT":
            return self.round_price_to_tick(position.entry_price * (Decimal("1") - profit_ratio))
        raise ConfigError("Cannot calculate take-profit price for a flat position.")

    def round_price_to_tick(self, price: Decimal) -> Decimal:
        assert self.symbol_rules is not None
        rounded = round_to_step(price, self.symbol_rules.tick_size)
        if rounded <= 0:
            raise ConfigError("Rounded trigger price must be greater than 0.")
        return rounded

    def calculate_tp1_quantity_from_original(self, original_quantity: Decimal) -> Decimal:
        assert self.symbol_rules is not None
        return round_to_step(original_quantity * self.config.tp1_close_ratio, self.symbol_rules.qty_step)

    def validate_close_quantity(self, quantity: Decimal, label: str) -> None:
        assert self.symbol_rules is not None
        if quantity < self.symbol_rules.min_qty:
            raise ConfigError(
                f"{label} close quantity {format_decimal(quantity)} is below the minimum lot size "
                f"{format_decimal(self.symbol_rules.min_qty)} for {self.config.symbol}."
            )

    def get_open_reduce_only_orders(self) -> list[dict[str, Any]]:
        orders = self.client.get_orders(symbol=self.config.symbol)
        return [order for order in orders if is_reduce_only_order(order)]

    def cancel_all_reduce_only_orders(self) -> None:
        if not self.config.can_place_real_orders:
            return
        for order in self.get_open_reduce_only_orders():
            self.cancel_order(order, reason="flat position cleanup")

    def cancel_order(self, order: dict[str, Any], reason: str) -> None:
        if not self.config.can_place_real_orders:
            self.log_order_block(f"canceling order for {reason}")
            return
        order_id = get_order_id(order)
        if order_id is None:
            return
        try:
            self.client.cancel_order(symbol=self.config.symbol, orderId=order_id)
            self.logger.info(
                "Canceled reduceOnly order. orderId=%s type=%s reason=%s",
                order_id,
                order.get("type"),
                reason,
            )
        except (ClientError, ServerError) as exc:
            self.logger.warning("Could not cancel orderId=%s: %s", order_id, exc)

    def initial_stop_reason(self, position: Position, mark_price: Decimal) -> Optional[str]:
        stop_loss_ratio = self.config.stop_loss_pct / Decimal("100")
        if position.side == "LONG":
            stop_price = position.entry_price * (Decimal("1") - stop_loss_ratio)
            if mark_price <= stop_price:
                return f"Long stop loss hit at {format_decimal(mark_price)}."
        if position.side == "SHORT":
            stop_price = position.entry_price * (Decimal("1") + stop_loss_ratio)
            if mark_price >= stop_price:
                return f"Short stop loss hit at {format_decimal(mark_price)}."
        return None

    def break_even_stop_reason(self, position: Position, mark_price: Decimal) -> Optional[str]:
        if self.exit_state is None or not self.exit_state.break_even_armed:
            return None
        if position.side == "LONG" and mark_price <= position.entry_price:
            return f"Long break-even stop hit at {format_decimal(mark_price)}."
        if position.side == "SHORT" and mark_price >= position.entry_price:
            return f"Short break-even stop hit at {format_decimal(mark_price)}."
        return None

    def tp2_reached(self, position: Position, mark_price: Decimal) -> bool:
        profit_pct = calculate_unrealized_profit_pct(position, mark_price)
        return profit_pct >= self.config.tp2_profit_pct

    def calculate_tp1_close_quantity(self, position: Position) -> Decimal:
        assert self.symbol_rules is not None
        assert self.exit_state is not None

        current_quantity = abs(position.quantity)
        target_quantity = round_to_step(
            self.exit_state.original_quantity * self.config.tp1_close_ratio,
            self.symbol_rules.qty_step,
        )
        quantity = min(target_quantity, current_quantity)
        if quantity < self.symbol_rules.min_qty:
            raise ConfigError(
                f"TP1 close quantity {format_decimal(quantity)} is below the minimum lot size "
                f"{format_decimal(self.symbol_rules.min_qty)} for {self.config.symbol}."
            )
        return quantity

    def handle_tp1_exit(
        self,
        position: Position,
        mark_price: Decimal,
        quantity: Decimal,
        reason: str,
        tp2_already_reached: bool,
    ) -> bool:
        if self.config.dry_run:
            self.logger.info(
                "Dry-run enabled. Planned TP1 partial close for %s position at %s. quantity=%s reason=%s",
                position.side,
                format_decimal(mark_price),
                format_decimal(quantity),
                reason,
            )
            if tp2_already_reached:
                self.logger.info(
                    "Dry-run enabled. TP2 is already reached, so the remaining position would also be closed."
                )
            return True

        self.logger.info(reason)
        self.close_position_quantity(
            position=position,
            quantity=quantity,
            exit_price=mark_price,
            reason=reason,
            final_exit=False,
        )
        assert self.exit_state is not None
        self.exit_state.tp1_done = True
        remaining_position = self.build_remaining_position_after_tp1(position, quantity)
        if remaining_position.is_open:
            self.arm_software_break_even(remaining_position)

        if tp2_already_reached:
            if remaining_position.is_open:
                tp2_reason = "TP2 was already reached after TP1. Closing remaining position."
                self.logger.info(tp2_reason)
                self.close_position(remaining_position, mark_price, tp2_reason)
        return True

    def build_remaining_position_after_tp1(
        self,
        position: Position,
        tp1_quantity: Decimal,
    ) -> Position:
        assert self.symbol_rules is not None

        remaining_quantity = round_to_step(
            abs(position.quantity) - tp1_quantity,
            self.symbol_rules.qty_step,
        )
        if remaining_quantity <= 0:
            return Position(quantity=Decimal("0"), entry_price=position.entry_price)
        signed_quantity = remaining_quantity if position.quantity > 0 else -remaining_quantity
        return Position(quantity=signed_quantity, entry_price=position.entry_price)

    def handle_full_exit(self, position: Position, mark_price: Decimal, reason: str) -> bool:
        if self.config.dry_run:
            self.logger.info(
                "Dry-run enabled. Planned full exit for %s position at %s. reason=%s",
                position.side,
                format_decimal(mark_price),
                reason,
            )
            return True

        self.logger.info(reason)
        self.close_position(position, mark_price, reason)
        return True

    def calculate_order_quantity(self, mark_price: Decimal) -> Decimal:
        assert self.symbol_rules is not None

        order_notional = self.calculate_order_notional_quote()
        raw_quantity = order_notional / mark_price
        quantity = round_to_step(raw_quantity, self.symbol_rules.qty_step)

        if quantity < self.symbol_rules.min_qty:
            raise ConfigError(
                f"Calculated quantity {format_decimal(quantity)} is below the minimum lot size "
                f"{format_decimal(self.symbol_rules.min_qty)} for {self.config.symbol}."
            )
        if quantity > self.symbol_rules.max_qty:
            raise ConfigError(
                f"Calculated quantity {format_decimal(quantity)} exceeds the maximum lot size "
                f"{format_decimal(self.symbol_rules.max_qty)} for {self.config.symbol}."
            )
        if self.symbol_rules.min_notional > 0 and quantity * mark_price < self.symbol_rules.min_notional:
            raise ConfigError(
                f"Order notional {format_decimal(quantity * mark_price)} is below Binance minimum "
                f"{format_decimal(self.symbol_rules.min_notional)}."
            )
        return quantity

    def calculate_order_notional_quote(self) -> Decimal:
        requested_notional = self.config.order_size_quote

        if not self.config.can_place_real_orders:
            self.logger.info(
                "Using fixed ORDER_SIZE from env=%s because live wallet sizing is not active.",
                format_decimal(requested_notional),
            )
            return requested_notional

        self.logger.info(
            "Position sizing: using requested ORDER_SIZE from env=%s for %s with leverage=%sx.",
            format_decimal(requested_notional),
            self.config.symbol,
            self.config.leverage,
        )
        return requested_notional

    def log_dry_run_entry_plan(self, action: str, quantity: Decimal, mark_price: Decimal) -> None:
        signed_quantity = quantity if action == "LONG" else -quantity
        planned_position = Position(quantity=signed_quantity, entry_price=mark_price)
        tp1_quantity = self.calculate_tp1_quantity_from_original(quantity)
        tp2_quantity = round_to_step(quantity - tp1_quantity, self.symbol_rules.qty_step)
        self.log_entry_risk_before_order(action, quantity, mark_price)
        self.logger.info(
            "Dry-run enabled. Planned entry=%s quantity=%s %s",
            action,
            format_decimal(quantity),
            self.config.symbol,
        )
        self.logger.info(
            "Dry-run protective orders: STOP_MARKET reduceOnly stop=%s quantity=%s; "
            "TP1 TAKE_PROFIT_MARKET reduceOnly stop=%s quantity=%s; "
            "TP2 TAKE_PROFIT_MARKET reduceOnly stop=%s quantity=%s.",
            format_decimal(self.calculate_stop_price(planned_position, break_even=False)),
            format_decimal(quantity),
            format_decimal(self.calculate_take_profit_price(planned_position, self.config.tp1_profit_pct)),
            format_decimal(tp1_quantity),
            format_decimal(self.calculate_take_profit_price(planned_position, self.config.tp2_profit_pct)),
            format_decimal(tp2_quantity),
        )

    def log_entry_risk_before_order(
        self,
        action: str,
        quantity: Decimal,
        estimated_entry_price: Decimal,
    ) -> None:
        signed_quantity = quantity if action == "LONG" else -quantity
        planned_position = Position(quantity=signed_quantity, entry_price=estimated_entry_price)
        stop_loss = self.calculate_stop_price(planned_position, break_even=False)
        tp1_price = self.calculate_take_profit_price(planned_position, self.config.tp1_profit_pct)
        tp2_price = self.calculate_take_profit_price(planned_position, self.config.tp2_profit_pct)
        notional = quantity * estimated_entry_price

        self.logger.info(
            "Entry plan before placing order: side=%s estimated_entry=%s stop_loss=%s "
            "tp1=%s tp2=%s position_size=%s %s notional_usdt=%s",
            action,
            format_decimal(estimated_entry_price),
            format_decimal(stop_loss),
            format_decimal(tp1_price),
            format_decimal(tp2_price),
            format_decimal(quantity),
            self.config.symbol,
            format_decimal(notional),
        )

    def build_trade_context_from_signal(self, signal: TradingSignal, position: Position) -> TradeContext:
        return TradeContext(
            trade_id=uuid4().hex,
            key=build_position_key(position),
            opened_at=datetime.now(timezone.utc),
            symbol=self.config.symbol,
            side=position.side,
            entry_price=position.entry_price,
            entry_reason=signal.reason,
            ema_fast=signal.ema_fast,
            ema_slow=signal.ema_slow,
            ema_trend=signal.ema_trend,
            ema_spread_pct=signal.ema_spread_pct,
            candle_body_ratio=signal.candle_body_ratio,
            distance_from_ema7_pct=signal.distance_from_ema7_pct,
            volume=signal.volume,
            previous_candle_direction=signal.previous_candle_direction,
            cooldown_status=self.peek_trade_block_reason() or "clear",
        )

    def build_recovered_trade_context(self, position: Position) -> TradeContext:
        self.logger.warning(
            "Recovered open %s position without original entry context. Trade CSV will mark entry_reason=recovered_existing_position.",
            position.side,
        )
        return TradeContext(
            trade_id=uuid4().hex,
            key=build_position_key(position),
            opened_at=datetime.now(timezone.utc),
            symbol=self.config.symbol,
            side=position.side,
            entry_price=position.entry_price,
            entry_reason="recovered_existing_position",
            ema_fast=Decimal("0"),
            ema_slow=Decimal("0"),
            ema_trend=Decimal("0"),
            ema_spread_pct=Decimal("0"),
            candle_body_ratio=Decimal("0"),
            distance_from_ema7_pct=Decimal("0"),
            volume=Decimal("0"),
            previous_candle_direction="unknown",
            cooldown_status=self.peek_trade_block_reason() or "clear",
        )

    def log_trade_to_csv(
        self,
        position: Position,
        exit_price: Decimal,
        quantity: Decimal,
        realized_pnl: Decimal,
        exit_reason: str,
        final_exit: bool,
    ) -> None:
        context = self.current_trade_context
        if context is None or context.key != build_position_key(position):
            context = self.build_recovered_trade_context(position)
            self.current_trade_context = context

        now = datetime.now(timezone.utc)
        tp1_hit_before_exit = context.tp1_hit_before_exit or "TP1" in exit_reason.upper()
        row = {
            "trade_id": context.trade_id,
            "timestamp": now.isoformat(),
            "symbol": context.symbol,
            "side": context.side,
            "entry_price": format_decimal(context.entry_price),
            "exit_price": format_decimal(exit_price),
            "quantity": format_decimal(quantity),
            "pnl_usdt": format_decimal(realized_pnl),
            "exit_reason": exit_reason,
            "entry_reason": context.entry_reason,
            "ema7": format_decimal(context.ema_fast),
            "ema25": format_decimal(context.ema_slow),
            "ema99": format_decimal(context.ema_trend),
            "ema_spread_pct": format_decimal(context.ema_spread_pct),
            "candle_body_ratio": format_decimal(context.candle_body_ratio),
            "distance_from_ema7_pct": format_decimal(context.distance_from_ema7_pct),
            "volume": format_decimal(context.volume),
            "previous_candle_direction": context.previous_candle_direction,
            "cooldown_status": context.cooldown_status,
            "tp1_hit_before_exit": str(tp1_hit_before_exit).lower(),
            "holding_time_seconds": format_decimal(Decimal(str((now - context.opened_at).total_seconds()))),
        }

        write_header = not TRADES_CSV_PATH.exists()
        with TRADES_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.logger.info(
            "Trade logged to %s. trade_id=%s pnl=%s exit_reason=%s",
            TRADES_CSV_PATH,
            context.trade_id,
            format_decimal(realized_pnl),
            exit_reason,
        )
        context.tp1_hit_before_exit = tp1_hit_before_exit
        if final_exit:
            self.current_trade_context = None

    def close_open_position_defensively(self, reason: str) -> None:
        if not self.config.can_place_real_orders:
            return

        try:
            position = self.get_position()
            self.last_known_position = position
        except Exception as exc:
            self.logger.exception("Could not fetch current position for defensive close.")
            send_discord_message(
                f"Defensive close warning\n"
                f"symbol: {self.config.symbol}\n"
                f"error: could not fetch current position: {exc}"
            )
            position = self.last_known_position

        if not position.is_open:
            self.logger.info("No live position found for defensive close.")
            return

        exit_price = self.safe_mark_price(position.entry_price)
        self.logger.error("Closing live position defensively. reason=%s", reason)
        send_discord_message(
            f"Defensive close\n"
            f"symbol: {self.config.symbol}\n"
            f"side: {position.side}\n"
            f"quantity: {format_decimal(abs(position.quantity))}\n"
            f"reason: {reason}"
        )
        try:
            self.close_position(position, exit_price, reason)
        except Exception:
            self.logger.exception("Defensive close failed. Manual intervention may be required.")
            send_discord_message(
                f"Defensive close failed\n"
                f"symbol: {self.config.symbol}\n"
                f"manual action may be required"
            )

    def close_expected_position_defensively(
        self,
        action: str,
        quantity: Decimal,
        fallback_exit_price: Decimal,
        reason: str,
    ) -> None:
        if not self.config.can_place_real_orders:
            return

        try:
            position = self.get_position()
            self.last_known_position = position
            if position.is_open:
                self.close_position(position, self.safe_mark_price(fallback_exit_price), reason)
                return
        except Exception as exc:
            self.logger.warning(
                "Could not confirm position before expected defensive close: %s. Sending reduceOnly close for expected quantity.",
                exc,
            )

        close_side = "SELL" if action == "LONG" else "BUY"
        self.submit_market_order(side=close_side, quantity=quantity, reduce_only=True)
        self.logger.error(
            "Sent defensive reduceOnly MARKET close for expected %s quantity=%s.",
            action,
            format_decimal(quantity),
        )
        send_discord_message(
            f"Defensive expected close sent\n"
            f"symbol: {self.config.symbol}\n"
            f"side: {action}\n"
            f"quantity: {format_decimal(quantity)}\n"
            f"reason: {reason}"
        )

    def safe_mark_price(self, fallback: Decimal) -> Decimal:
        try:
            return self.get_mark_price()
        except Exception as exc:
            self.logger.warning(
                "Could not fetch mark price for defensive close; using fallback %s. error=%s",
                format_decimal(fallback),
                exc,
            )
            return fallback

    def close_position(self, position: Position, exit_price: Decimal, reason: str) -> None:
        quantity = round_to_step(abs(position.quantity), self.symbol_rules.qty_step)
        if quantity <= 0:
            self.logger.info("No open quantity to close.")
            return

        self.close_position_quantity(
            position=position,
            quantity=quantity,
            exit_price=exit_price,
            reason=reason,
            final_exit=True,
        )

    def close_position_quantity(
        self,
        position: Position,
        quantity: Decimal,
        exit_price: Decimal,
        reason: str,
        final_exit: bool,
    ) -> None:
        side = "SELL" if position.side == "LONG" else "BUY"
        self.logger.info(
            "Closing %s of %s position with %s %s.",
            "all" if final_exit else "part",
            position.side,
            format_decimal(quantity),
            self.config.symbol,
        )
        self.submit_market_order(side=side, quantity=quantity, reduce_only=True)
        realized_pnl = calculate_realized_pnl(position, exit_price, quantity)
        self.record_trade_close(realized_pnl)
        self.log_trade_to_csv(
            position=position,
            exit_price=exit_price,
            quantity=quantity,
            realized_pnl=realized_pnl,
            exit_reason=reason,
            final_exit=final_exit,
        )
        if final_exit:
            self.cancel_all_reduce_only_orders()
            self.exit_state = None
        exit_message = (
            f"{'Exit' if final_exit else 'Partial exit'} executed\n"
            f"symbol: {self.config.symbol}\n"
            f"side: {position.side}\n"
            f"reason: {reason}\n"
            f"exit_price: {format_decimal(exit_price)}\n"
            f"closed_quantity: {format_decimal(quantity)}\n"
            f"estimated_pnl_usdt: {format_decimal(realized_pnl)}\n"
            f"trades_today: {self.trades_today}\n"
            f"losses_today: {self.losses_today}"
        )
        self.logger.info(exit_message.replace("\n", " | "))
        send_discord_message(exit_message)
        if (
            final_exit
            and self.config.max_trades_per_day > 0
            and self.trades_today >= self.config.max_trades_per_day
        ):
            self.run_learning_analysis(f"max_trades_per_day:{self.current_day}")

    def open_position(
        self,
        signal: TradingSignal,
        quantity: Decimal,
        mark_price: Decimal,
    ) -> None:
        action = signal.action
        side = "BUY" if action == "LONG" else "SELL"
        self.logger.info(
            "Opening %s position with %s %s.",
            action,
            format_decimal(quantity),
            self.config.symbol,
        )
        self.log_entry_risk_before_order(action, quantity, mark_price)
        self.submit_market_order(side=side, quantity=quantity, reduce_only=False)

        try:
            live_position = self.wait_for_live_position(action)
            self.last_known_position = live_position
            self.current_trade_context = self.build_trade_context_from_signal(signal, live_position)
            self.install_software_managed_exit_state(live_position)
        except Exception as exc:
            self.logger.exception("Software protection setup failed after entry. Closing position defensively.")
            send_discord_message(
                f"Entry software protection failed\n"
                f"symbol: {self.config.symbol}\n"
                f"side: {action}\n"
                f"error: {exc}\n"
                f"action: closing position defensively"
            )
            self.close_expected_position_defensively(
                action=action,
                quantity=quantity,
                fallback_exit_price=mark_price,
                reason="Software protection setup failed after entry.",
            )
            raise

        self.trades_today += 1
        entry_message = (
            f"Entry executed\n"
            f"symbol: {self.config.symbol}\n"
            f"side: {action}\n"
            f"price: {format_decimal(live_position.entry_price or mark_price)}\n"
            f"quantity: {format_decimal(abs(live_position.quantity))}\n"
            f"reason: {signal.reason}\n"
            f"protection: software-managed SL/TP active\n"
            f"trades_today: {self.trades_today}\n"
            f"mode: {self.config.mode_label}"
        )
        self.logger.info(entry_message.replace("\n", " | "))
        send_discord_message(entry_message)

    def wait_for_live_position(self, expected_side: str) -> Position:
        for _ in range(5):
            position = self.get_position()
            if position.is_open and position.side == expected_side:
                return position
            time.sleep(0.5)
        raise RuntimeError(f"Entry order did not produce an open {expected_side} position.")

    def submit_market_order(self, side: str, quantity: Decimal, reduce_only: bool) -> dict[str, Any]:
        self.require_real_order_permission("placing market order")
        order_params: dict[str, Any] = {
            "symbol": self.config.symbol,
            "side": side,
            "type": "MARKET",
            "quantity": format_decimal(quantity),
        }
        if reduce_only:
            order_params["reduceOnly"] = "true"

        order = self.client.new_order(**order_params)
        self.logger.info(
            "Order submitted. orderId=%s status=%s side=%s quantity=%s",
            order.get("orderId"),
            order.get("status"),
            order.get("side"),
            order.get("origQty"),
        )
        return order

    def previous_candle_touched_ema(self, previous: pd.Series, side: str) -> bool:
        if side == "LONG":
            return previous["low"] <= previous["ema_fast"] or previous["low"] <= previous["ema_slow"]
        if side == "SHORT":
            return previous["high"] >= previous["ema_fast"] or previous["high"] >= previous["ema_slow"]
        return False

    def build_hold_reason(
        self,
        latest: pd.Series,
        previous: pd.Series,
        bullish_latest: bool,
        bearish_latest: bool,
        long_previous_touch: bool,
        short_previous_touch: bool,
    ) -> str:
        reasons: list[str] = []
        if latest["close"] <= latest["ema_slow"]:
            reasons.append("Latest close is not above EMA25 for a long setup.")
        if latest["close"] >= latest["ema_slow"]:
            reasons.append("Latest close is not below EMA25 for a short setup.")

        if latest["ema_fast"] == latest["ema_slow"]:
            reasons.append("EMA7 and EMA25 are flat against each other.")
        elif latest["ema_fast"] > latest["ema_slow"] and not bullish_latest:
            reasons.append("Bullish trend exists but the latest closed candle is not bullish.")
        elif latest["ema_fast"] < latest["ema_slow"] and not bearish_latest:
            reasons.append("Bearish trend exists but the latest closed candle is not bearish.")

        if not long_previous_touch and not short_previous_touch:
            reasons.append("Previous candle did not strictly touch EMA7 or EMA25.")
        if latest["close"] > latest["ema_slow"] and latest["close"] <= previous["close"]:
            reasons.append("Current close is not above the previous close for long confirmation.")
        if latest["close"] < latest["ema_slow"] and latest["close"] >= previous["close"]:
            reasons.append("Current close is not below the previous close for short confirmation.")
        if not reasons:
            reasons.append("Strict closed-candle confirmation conditions were not fully met.")
        return " ".join(reasons[:3])

    def maybe_send_signal_alert(self, signal: TradingSignal, mark_price: Decimal) -> None:
        signal_key = f"{signal.candle_close_time.isoformat()}:{signal.action}"
        if self.last_signal_alert_key == signal_key:
            return
        self.last_signal_alert_key = signal_key
        message = (
            f"Signal detected\n"
            f"symbol: {self.config.symbol}\n"
            f"action: {signal.action}\n"
            f"close_price: {format_decimal(signal.close_price)}\n"
            f"mark_price: {format_decimal(mark_price)}\n"
            f"ema7: {format_decimal(signal.ema_fast)}\n"
            f"ema25: {format_decimal(signal.ema_slow)}\n"
            f"ema99: {format_decimal(signal.ema_trend)}\n"
            f"reason: {signal.reason}\n"
            f"mode: {self.config.mode_label}"
        )
        send_discord_message(message)

    def maybe_send_heartbeat(self, mark_price: Decimal, position: Position) -> None:
        heartbeat_seconds = self.config.heartbeat_interval_minutes * 60
        elapsed = time.monotonic() - self.last_heartbeat_at
        if elapsed < heartbeat_seconds:
            return
        send_discord_message(
            self.build_status_message(
                event="Heartbeat",
                status=self.current_status(),
                mark_price=mark_price,
                position=position,
            )
        )
        self.last_heartbeat_at = time.monotonic()

    def build_status_message(
        self,
        event: str,
        status: str,
        mark_price: Decimal,
        position: Position,
    ) -> str:
        lines = [
            event,
            f"status: {status}",
            f"mode: {self.config.mode_label}",
            f"symbol: {self.config.symbol}",
            f"mark price: {format_decimal(mark_price)}",
            f"open position side: {position.side}",
            f"trades_today: {self.trades_today}",
            f"losses_today: {self.losses_today}",
            f"last signal: {self.last_signal}",
            f"dry-run/live status: {self.config.mode_label}",
        ]
        limit_reason = self.peek_trade_block_reason()
        if limit_reason:
            lines.append(f"trade lock: {limit_reason}")
        return "\n".join(lines)

    def current_status(self) -> str:
        return "TRADE_LIMIT_HIT" if self.peek_trade_block_reason() else "RUNNING"

    def log_order_block(self, action: str) -> None:
        reason = self.config.order_block_reason or "unknown safety gate"
        self.logger.warning(
            "Refusing real order action (%s): %s.",
            action,
            reason,
        )

    def require_real_order_permission(self, action: str) -> None:
        if self.config.can_place_real_orders:
            return
        reason = self.config.order_block_reason or "unknown safety gate"
        self.logger.warning("Refusing real order action (%s): %s.", action, reason)
        raise ConfigError(f"Real order action blocked: {reason}.")

    def reset_daily_counters_if_needed(self) -> None:
        today = utc_today()
        if today == self.current_day:
            return
        previous_day = self.current_day
        self.run_learning_analysis(f"utc_day_rollover:{previous_day}")
        self.current_day = today
        self.trades_today = 0
        self.losses_today = 0
        self.daily_loss_usdt = Decimal("0")
        self.daily_profit_usdt = Decimal("0")
        self.trade_shutdown_reason = None
        self.trade_shutdown_notified = False
        self.last_signal_alert_key = None
        self.logger.info("New UTC day detected. Daily counters have been reset.")

    def peek_trade_block_reason(self) -> Optional[str]:
        if self.config.max_trades_per_day > 0 and self.trades_today >= self.config.max_trades_per_day:
            return (
                f"MAX_TRADES_PER_DAY reached "
                f"({self.trades_today}/{self.config.max_trades_per_day})."
            )
        if self.losses_today >= self.config.stop_after_losses:
            return (
                f"STOP_AFTER_LOSSES reached "
                f"({self.losses_today}/{self.config.stop_after_losses})."
            )
        if self.daily_loss_usdt >= self.config.max_daily_loss_usdt:
            return (
                f"MAX_DAILY_LOSS_USDT reached "
                f"({format_decimal(self.daily_loss_usdt)}/{format_decimal(self.config.max_daily_loss_usdt)})."
            )
        if self.daily_profit_usdt >= self.config.max_daily_profit_usdt:
            return (
                f"MAX_DAILY_PROFIT_USDT reached "
                f"({format_decimal(self.daily_profit_usdt)}/{format_decimal(self.config.max_daily_profit_usdt)})."
            )
        return None

    def handle_trade_block(
        self,
        reason: str,
        action: str,
        mark_price: Decimal,
        position: Position,
    ) -> None:
        self.logger.warning("Refusing new %s trade because of daily limits: %s", action, reason)
        if self.trade_shutdown_reason != reason:
            self.trade_shutdown_reason = reason
            self.trade_shutdown_notified = False
        if self.trade_shutdown_notified:
            return
        send_discord_message(
            self.build_status_message(
                event="Trading shutdown due to daily limit",
                status="TRADE_LIMIT_HIT",
                mark_price=mark_price,
                position=position,
            )
            + f"\nreason: {reason}"
        )
        self.trade_shutdown_notified = True
        if reason.startswith("MAX_TRADES_PER_DAY"):
            self.run_learning_analysis(f"max_trades_per_day:{self.current_day}")

    def record_trade_close(self, realized_pnl: Decimal) -> None:
        if realized_pnl >= 0:
            self.daily_profit_usdt += realized_pnl
        else:
            self.daily_loss_usdt += abs(realized_pnl)
            self.losses_today += 1
        self.logger.info(
            "Daily performance updated. profit=%s loss=%s losses_today=%s",
            format_decimal(self.daily_profit_usdt),
            format_decimal(self.daily_loss_usdt),
            self.losses_today,
        )

    def run_learning_analysis(self, trigger: str) -> None:
        if trigger in self.learning_analysis_triggers_run:
            return
        self.learning_analysis_triggers_run.add(trigger)

        if self.config.auto_apply_learning:
            self.warn_learning_auto_apply_if_needed()

        if not TRADE_ANALYZER_PATH.exists():
            self.logger.warning("Learning analyzer was not found at %s.", TRADE_ANALYZER_PATH)
            return

        self.logger.info("Running suggestion-only learning analysis. trigger=%s", trigger)
        env = os.environ.copy()
        env["MIN_TRADES_BEFORE_LEARNING"] = str(self.config.min_trades_before_learning)
        env["AUTO_APPLY_LEARNING"] = "true" if self.config.auto_apply_learning else "false"

        try:
            result = subprocess.run(
                [sys.executable, str(TRADE_ANALYZER_PATH)],
                cwd=PROJECT_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
        except Exception as exc:
            self.logger.warning("Learning analysis could not run: %s", exc)
            send_discord_message(f"Learning analysis failed\ntrigger: {trigger}\nerror: {exc}")
            return

        if result.stdout.strip():
            self.logger.info("Learning analyzer output: %s", result.stdout.strip())
        if result.stderr.strip():
            self.logger.warning("Learning analyzer stderr: %s", result.stderr.strip())
        if result.returncode != 0:
            self.logger.warning("Learning analyzer exited with code %s.", result.returncode)
            send_discord_message(
                f"Learning analysis failed\ntrigger: {trigger}\nexit_code: {result.returncode}"
            )
            return

        send_discord_message(self.build_learning_discord_summary(trigger))

    def build_learning_discord_summary(self, trigger: str) -> str:
        if not LEARNING_STATE_PATH.exists():
            return (
                "Learning analysis completed\n"
                f"trigger: {trigger}\n"
                "suggested_only: true\n"
                "summary: learning_state.json was not found"
            )

        try:
            state = json.loads(LEARNING_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return (
                "Learning analysis completed\n"
                f"trigger: {trigger}\n"
                "suggested_only: true\n"
                f"summary: could not read learning_state.json: {exc}"
            )

        metrics = state.get("metrics", {})
        recommendations = state.get("recommended_parameter_changes", [])
        recommendation_text = "none"
        if recommendations:
            recommendation_text = "; ".join(
                str(item.get("parameter", "filter")) for item in recommendations[:3]
            )
        return (
            "Learning analysis completed\n"
            f"trigger: {trigger}\n"
            f"suggested_only: {state.get('suggested_only', True)}\n"
            f"total_trades: {metrics.get('total_trades', 0)}\n"
            f"win_rate: {metrics.get('win_rate_pct', 0)}%\n"
            f"profit_factor: {metrics.get('profit_factor', 0)}\n"
            f"top_suggestions: {recommendation_text}\n"
            "report: learning_report.md"
        )


def build_position_key(position: Position) -> str:
    return (
        f"{position.side}:"
        f"{format_decimal(position.entry_price)}"
    )


def get_order_id(order: dict[str, Any]) -> Optional[int]:
    raw_order_id = order.get("orderId")
    if raw_order_id is None:
        return None
    try:
        return int(raw_order_id)
    except (TypeError, ValueError):
        return None


def is_reduce_only_order(order: dict[str, Any]) -> bool:
    reduce_only = order.get("reduceOnly")
    return reduce_only is True or str(reduce_only).lower() == "true"


def format_binance_api_error(exc: ClientError) -> str:
    code = getattr(exc, "error_code", None) or getattr(exc, "code", "unknown")
    message = getattr(exc, "error_message", None) or getattr(exc, "message", None) or str(exc)
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return f"status_code={status_code} code={code} message={message}"
    return f"code={code} message={message}"


def calculate_unrealized_profit_pct(position: Position, mark_price: Decimal) -> Decimal:
    if position.entry_price <= 0:
        return Decimal("0")
    if position.side == "LONG":
        return ((mark_price - position.entry_price) / position.entry_price) * Decimal("100")
    if position.side == "SHORT":
        return ((position.entry_price - mark_price) / position.entry_price) * Decimal("100")
    return Decimal("0")


def calculate_realized_pnl(
    position: Position,
    exit_price: Decimal,
    quantity: Optional[Decimal] = None,
) -> Decimal:
    closed_quantity = quantity or abs(position.quantity)
    if position.side == "LONG":
        return (exit_price - position.entry_price) * closed_quantity
    if position.side == "SHORT":
        return (position.entry_price - exit_price) * closed_quantity
    return Decimal("0")


def send_discord_message(content: str) -> None:
    webhook_url = normalize_optional_value(os.getenv("DISCORD_WEBHOOK_URL"))
    if not webhook_url or not content:
        return
    try:
        response = requests.post(
            webhook_url,
            json=build_discord_webhook_payload(content),
            timeout=DISCORD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.getLogger("discord").warning("Discord notification failed: %s", exc)


def build_discord_webhook_payload(content: str) -> dict[str, Any]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = lines[0] if lines else "Bot update"
    fields: list[dict[str, Any]] = []
    description_lines: list[str] = []

    for line in lines[1:]:
        if ":" not in line:
            description_lines.append(line)
            continue

        raw_name, raw_value = line.split(":", 1)
        name = raw_name.strip()
        value = raw_value.strip()
        if not name or not value:
            description_lines.append(line)
            continue

        if len(fields) >= DISCORD_MAX_EMBED_FIELDS:
            description_lines.append(line)
            continue

        fields.append(
            {
                "name": format_discord_field_name(name),
                "value": format_discord_field_value(name, value),
                "inline": is_discord_inline_field(name),
            }
        )

    description = "\n".join(description_lines).strip()
    embed: dict[str, Any] = {
        "title": truncate_discord_value(title, 256),
        "color": discord_color_for_message(title, fields, description),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author": {
            "name": "XRPUSDC Scalping Bot",
        },
        "footer": {
            "text": "Binance Futures bot | alerts are informational, learning is suggestion-only",
        },
    }
    if description:
        embed["description"] = truncate_discord_value(description, DISCORD_MAX_DESCRIPTION_LENGTH)
    if fields:
        embed["fields"] = fields

    return {
        "username": "XRPUSDC Scalping Bot",
        "allowed_mentions": {"parse": []},
        "embeds": [embed],
    }


def discord_color_for_message(
    title: str,
    fields: list[dict[str, Any]],
    description: str,
) -> int:
    haystack = " ".join(
        [
            title,
            description,
            " ".join(str(field.get("value", "")) for field in fields),
        ]
    ).lower()

    if any(keyword in haystack for keyword in ("failed", "error", "defensive", "blocked", "shutdown")):
        return DISCORD_EMBED_COLORS["danger"]
    if any(keyword in haystack for keyword in ("warning", "disabled", "limit", "stop hit")):
        return DISCORD_EMBED_COLORS["warning"]
    if "learning" in haystack or "suggested_only" in haystack:
        return DISCORD_EMBED_COLORS["learning"]
    if any(keyword in haystack for keyword in ("entry executed", "tp1", "tp2", "profit", "completed")):
        return DISCORD_EMBED_COLORS["success"]
    if any(keyword in haystack for keyword in ("signal detected", "heartbeat", "bot started", "running")):
        return DISCORD_EMBED_COLORS["info"]
    if "exit" in haystack or "loss" in haystack:
        return DISCORD_EMBED_COLORS["warning"]
    return DISCORD_EMBED_COLORS["neutral"]


def format_discord_field_name(name: str) -> str:
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return "Detail"
    return cleaned.title()


def is_discord_inline_field(name: str) -> bool:
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    block_fields = {
        "reason",
        "entry_reason",
        "exit_reason",
        "last_signal",
        "summary",
        "error",
        "action",
        "trade_lock",
        "top_suggestions",
    }
    return normalized not in block_fields


def format_discord_field_value(name: str, value: str) -> str:
    truncated = truncate_discord_value(value, DISCORD_MAX_FIELD_VALUE_LENGTH)
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    plain_text_fields = {
        "reason",
        "entry_reason",
        "exit_reason",
        "last_signal",
        "summary",
        "error",
        "action",
        "trade_lock",
        "top_suggestions",
    }
    if normalized in plain_text_fields or "\n" in truncated or len(truncated) > 90:
        return truncated
    return f"`{truncated}`"


def truncate_discord_value(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def decimal_from_number(value: Any) -> Decimal:
    return Decimal(str(round(float(value), 8)))


def safe_decimal_ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    return numerator / denominator


def candle_direction(candle: pd.Series) -> str:
    if candle["close"] > candle["open"]:
        return "bullish"
    if candle["close"] < candle["open"]:
        return "bearish"
    return "doji"


def format_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") or "0"


def normalize_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip()
    placeholders = {
        "YOUR_BINANCE_API_KEY",
        "YOUR_BINANCE_API_SECRET",
        "REPLACE_ME",
        "CHANGE_ME",
        "YOUR_KEY_HERE",
        "YOUR_SECRET_HERE",
    }
    if cleaned.upper() in placeholders:
        return None
    return cleaned


def normalize_optional_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def get_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value.")


def get_order_size_quote_env(default: str) -> Decimal:
    if os.getenv("ORDER_SIZE_USDC") and os.getenv("ORDER_SIZE_USDC", "").strip():
        return get_decimal_env("ORDER_SIZE_USDC", default)
    return get_decimal_env("ORDER_SIZE_USDT", default)


def get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc


def get_decimal_env(name: str, default: str) -> Decimal:
    raw_value = os.getenv(name)
    value = raw_value.strip() if raw_value and raw_value.strip() else default
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ConfigError(f"{name} must be a valid decimal number.") from exc


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        raise ConfigError(f"Unsupported LOG_LEVEL '{level}'.")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BOT_LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def main() -> int:
    try:
        config = BotConfig.from_env()
        configure_logging(config.log_level)
        if not config.has_credentials:
            logging.getLogger("main").warning(
                "API credentials are missing or placeholders are still present. "
                "The bot will only run safely in dry-run mode."
            )

        bot = BinanceFuturesBot(config)
        bot.run()
        return 0
    except KeyboardInterrupt:
        logging.getLogger("main").info("Execution interrupted by user.")
        return 130
    except (ConfigError, ClientError, ServerError, requests.RequestException) as exc:
        logging.getLogger("main").error("%s", exc)
        send_discord_message(f"Bot error\n{exc}")
        return 1
    except Exception:
        logging.getLogger("main").exception("Unexpected error while running the bot.")
        send_discord_message("Bot error\nUnexpected error while running the bot. Check logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
