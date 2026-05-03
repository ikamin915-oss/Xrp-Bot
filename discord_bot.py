#!/usr/bin/env python3
"""MoneyMaker Discord control bot for the XRPUSDC Futures project.

Run this beside main.py on the same VPS. Commands are admin-only through
ALLOWED_DISCORD_USER_IDS. The bot can pause/resume entries, restart the local
trading process, run suggestion-only analysis, and close an existing position
with a reduceOnly MARKET order after confirmation.

It cannot change leverage, order size, strategy rules, .env, or withdraw funds.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional

import discord
from binance.error import ClientError, ServerError
from binance.um_futures import UMFutures
from discord.ext import commands
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
BACKUPS_DIR = PROJECT_DIR / "backups"
MAIN_BOT_PATH = PROJECT_DIR / "main.py"
TRADE_ANALYZER_PATH = PROJECT_DIR / "trade_analyzer.py"
BOT_LOG_PATH = PROJECT_DIR / "bot.log"
TRADES_CSV_PATH = PROJECT_DIR / "trades.csv"
LEARNING_REPORT_PATH = PROJECT_DIR / "learning_report.md"
LEARNING_STATE_PATH = PROJECT_DIR / "learning_state.json"
SUGGESTED_ENV_UPDATE_PATH = PROJECT_DIR / "suggested_env_update.txt"
SUGGESTED_STRATEGY_UPDATE_PATH = PROJECT_DIR / "suggested_strategy_update.md"
PAUSED_LOCK_PATH = PROJECT_DIR / "PAUSED.lock"
BOT_LOCK_PATH = PROJECT_DIR / "bot.lock"
ADMIN_ACTION_LOG_PATH = PROJECT_DIR / "admin_actions.log"
FUTURES_LIVE_URL = "https://fapi.binance.me"
FUTURES_TESTNET_URL = "https://demo-fapi.binance.com"
TRADE_SCREEN_NAME = "xrp-trade-bot"
DISCORD_SCREEN_NAME = "xrp-discord-bot"
MAX_FIELD_LENGTH = 1024
MAX_DESCRIPTION_LENGTH = 3900
DEFAULT_TAIL_LINES = 80
CLOSE_CONFIRM_SECONDS = 30
UPGRADE_APPROVAL_SECONDS = 10 * 60
UPGRADE_CHECK_SECONDS = 60
PROTECTED_ENV_KEY_FRAGMENTS = (
    "API",
    "KEY",
    "SECRET",
    "TOKEN",
    "WEBHOOK",
    "PASSWORD",
    "BINANCE",
    "DISCORD",
)
SIZE_ENV_KEYS = {
    "LEVERAGE",
    "ORDER_SIZE_USDT",
    "ORDER_SIZE_USDC",
    "WALLET_MARGIN_PCT",
}
SAFE_ENV_UPGRADE_KEYS = {
    "MIN_EMA_SPREAD_PCT",
    "MIN_CANDLE_BODY_RATIO",
    "MAX_DISTANCE_FROM_EMA7_PCT",
    "COOLDOWN_AFTER_LOSS_CANDLES",
    "MIN_VOLUME_FILTER",
    "CHOP_FILTER",
    "AVOID_BIG_CANDLE_LATE_ENTRY",
    "EMA_SLOPE_LOOKBACK",
    "EMA99_SLOPE_LOOKBACK",
    "MIN_EMA_FAST_SLOPE_PCT",
    "MIN_EMA_SLOW_SLOPE_PCT",
    "MIN_EMA99_SLOPE_PCT",
    "TREND_MOMENTUM_LOOKBACK",
    "MIN_TREND_MOMENTUM_PCT",
}


@dataclass(frozen=True)
class DiscordBotConfig:
    token: str
    prefix: str
    allowed_channel_id: Optional[int]
    allowed_user_ids: set[int]
    status_tail_lines: int
    enable_control: bool
    allow_close_position: bool
    allow_hardkill: bool
    api_key: Optional[str]
    api_secret: Optional[str]
    symbol: str
    use_testnet: bool
    request_timeout: int
    auto_upgrade_enabled: bool
    auto_upgrade_interval_hours: int
    upgrade_confirm_password: Optional[str]
    auto_code_upgrade_enabled: bool

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def futures_base_url(self) -> str:
        return FUTURES_TESTNET_URL if self.use_testnet else FUTURES_LIVE_URL


@dataclass(frozen=True)
class PositionSnapshot:
    side: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def abs_quantity(self) -> Decimal:
        return abs(self.quantity)


def load_config() -> DiscordBotConfig:
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    token = normalize_optional_value(os.getenv("DISCORD_BOT_TOKEN"))
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing in .env.")

    admin_ids = os.getenv("ALLOWED_DISCORD_USER_IDS") or os.getenv("DISCORD_ALLOWED_USER_IDS")
    return DiscordBotConfig(
        token=token,
        prefix=normalize_optional_value(os.getenv("DISCORD_COMMAND_PREFIX")) or "!",
        allowed_channel_id=parse_optional_int(os.getenv("DISCORD_ALLOWED_CHANNEL_ID")),
        allowed_user_ids=parse_id_set(admin_ids),
        status_tail_lines=parse_positive_int(os.getenv("DISCORD_STATUS_TAIL_LINES"), DEFAULT_TAIL_LINES),
        enable_control=parse_bool(os.getenv("ENABLE_DISCORD_CONTROL"), True),
        allow_close_position=parse_bool(os.getenv("ALLOW_DISCORD_CLOSE_POSITION"), True),
        allow_hardkill=parse_bool(os.getenv("ALLOW_DISCORD_HARDKILL"), True),
        api_key=normalize_secret(os.getenv("BINANCE_API_KEY")),
        api_secret=normalize_secret(os.getenv("BINANCE_API_SECRET")),
        symbol=(normalize_optional_value(os.getenv("SYMBOL")) or "XRPUSDC").upper(),
        use_testnet=parse_bool(os.getenv("USE_TESTNET"), True),
        request_timeout=parse_positive_int(os.getenv("REQUEST_TIMEOUT"), 20),
        auto_upgrade_enabled=parse_bool(os.getenv("AUTO_UPGRADE_ENABLED"), False),
        auto_upgrade_interval_hours=parse_positive_int(os.getenv("AUTO_UPGRADE_INTERVAL_HOURS"), 12),
        upgrade_confirm_password=normalize_optional_value(os.getenv("UPGRADE_CONFIRM_PASSWORD")),
        auto_code_upgrade_enabled=parse_bool(os.getenv("AUTO_CODE_UPGRADE_ENABLED"), False),
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_bot(config: DiscordBotConfig) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=config.prefix, intents=intents, help_command=None)
    pending_close_confirmations: dict[int, float] = {}
    upgrade_state: dict[str, Any] = new_upgrade_state()

    async def ensure_admin(ctx: commands.Context) -> bool:
        if config.allowed_channel_id and ctx.guild is not None and ctx.channel.id != config.allowed_channel_id:
            return False
        if not config.enable_control:
            await ctx.reply(
                embed=make_embed("Discord control disabled", "ENABLE_DISCORD_CONTROL is false.", discord.Color.orange()),
                mention_author=False,
            )
            return False
        if not config.allowed_user_ids:
            await ctx.reply(
                embed=make_embed(
                    "No Discord admins configured",
                    "Set ALLOWED_DISCORD_USER_IDS in .env before using control commands.",
                    discord.Color.red(),
                ),
                mention_author=False,
            )
            return False
        if ctx.author.id not in config.allowed_user_ids:
            await ctx.reply(
                embed=make_embed("Access denied", "This Discord user is not allowed to control MoneyMaker.", discord.Color.red()),
                mention_author=False,
            )
            return False
        return True

    @bot.event
    async def on_ready() -> None:
        logging.getLogger("discord_bot").info(
            "MoneyMaker Discord bot online as %s. prefix=%s control=%s admins=%s",
            bot.user,
            config.prefix,
            config.enable_control,
            len(config.allowed_user_ids),
        )
        if not getattr(bot, "_money_maker_upgrade_task_started", False):
            bot._money_maker_upgrade_task_started = True
            bot.loop.create_task(auto_upgrade_loop(bot, config, upgrade_state))

    @bot.event
    async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        logging.getLogger("discord_bot").warning("Command error: %s", error)
        await ctx.reply(
            embed=make_embed("Command failed", truncate(str(error), 1200), discord.Color.red()),
            mention_author=False,
        )

    @bot.command(name="help")
    async def help_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        embed = make_embed(
            "MoneyMaker admin commands",
            (
                f"`{config.prefix}status` - show bot status and local lock files\n"
                f"`{config.prefix}position` - show open Binance position\n"
                f"`{config.prefix}pnl` - show PnL summary from trades.csv\n"
                f"`{config.prefix}report` - show latest learning_report.md summary\n"
                f"`{config.prefix}analyze` - run suggestion-only trade_analyzer.py\n"
                f"`{config.prefix}upgrade_status` - show pending auto-upgrade state\n"
                f"`{config.prefix}show_suggestions` - show proposed safe .env changes\n"
                f"`{config.prefix}approve_upgrade PASSWORD` - approve within 10 minutes\n"
                f"`{config.prefix}reject_upgrade` - reject pending upgrade\n"
                f"`{config.prefix}pause` / `{config.prefix}resume` - block or allow new entries\n"
                f"`{config.prefix}close` - ask for reduceOnly close confirmation\n"
                f"`{config.prefix}confirmclose` - confirm close within {CLOSE_CONFIRM_SECONDS} seconds\n"
                f"`{config.prefix}hardkill` - stop main.py/analyzer, not this Discord bot\n"
                f"`{config.prefix}restart` - restart main.py using screen or subprocess"
            ),
            discord.Color.blurple(),
        )
        embed.set_footer(text="Admin-only. No leverage/order-size/strategy mutation commands exist.")
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="alive")
    async def alive_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        await ctx.reply(
            embed=make_embed("MoneyMaker is online", "Discord control is active and admin-gated.", discord.Color.green()),
            mention_author=False,
        )

    @bot.command(name="status")
    async def status_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "status", "status requested")
        lines = tail_lines(BOT_LOG_PATH, config.status_tail_lines)
        embed = make_embed("Trading bot status", color=discord.Color.blue())
        embed.add_field(name="Project", value=f"`{PROJECT_DIR}`", inline=False)
        embed.add_field(name="Mode", value="`admin-only control`", inline=True)
        embed.add_field(name="Symbol", value=f"`{config.symbol}`", inline=True)
        embed.add_field(name="Futures URL", value=f"`{config.futures_base_url}`", inline=False)
        embed.add_field(name="bot.lock", value=file_status(BOT_LOCK_PATH), inline=True)
        embed.add_field(name="PAUSED.lock", value=file_status(PAUSED_LOCK_PATH), inline=True)
        embed.add_field(name="bot.log", value=file_status(BOT_LOG_PATH), inline=True)
        embed.add_field(name="Last signal", value=find_last_log_event(lines, ("Signal=", "PAUSED.lock")) or "No signal found.", inline=False)
        embed.add_field(name="Position log", value=find_last_log_event(lines, ("Current position:",)) or "No position line found.", inline=False)
        embed.add_field(
            name="Last trade event",
            value=find_last_log_event(lines, ("Entry executed", "Exit executed", "Partial exit executed")) or "No trade event found.",
            inline=False,
        )
        embed.add_field(
            name="Last warning/error",
            value=find_last_log_event(lines, ("WARNING", "ERROR", "CRITICAL", "Bot crashed")) or "No recent warning/error found.",
            inline=False,
        )
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="position")
    async def position_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "position", "position requested")
        try:
            snapshot = await asyncio.to_thread(fetch_position_snapshot, config)
        except Exception as exc:
            await ctx.reply(
                embed=make_embed("Position check failed", format_binance_error(exc), discord.Color.red()),
                mention_author=False,
            )
            return
        await ctx.reply(embed=build_position_embed(snapshot, config.symbol), mention_author=False)

    @bot.command(name="pnl")
    async def pnl_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "pnl", "pnl requested")
        await ctx.reply(embed=build_pnl_embed(), mention_author=False)

    @bot.command(name="report")
    async def report_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "report", "learning report requested")
        await ctx.reply(embed=build_report_embed(), mention_author=False)

    @bot.command(name="analyze")
    async def analyze_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "analyze", "manual learning analysis requested")
        await ctx.reply(
            embed=make_embed("Learning analysis started", "Running trade_analyzer.py in suggestion-only mode.", discord.Color.teal()),
            mention_author=False,
        )
        result = await run_analyzer_subprocess()
        if result["returncode"] != 0:
            await ctx.send(
                embed=make_embed(
                    "Learning analysis failed",
                    truncate(result.get("stderr") or result.get("stdout") or "No output.", 1600),
                    discord.Color.red(),
                )
            )
            return
        await ctx.send(embed=build_learning_state_embed("Learning analysis completed"))

    @bot.command(name="upgrade_status")
    async def upgrade_status_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "upgrade_status", "upgrade status requested")
        await ctx.reply(embed=build_upgrade_status_embed(upgrade_state, config), mention_author=False)

    @bot.command(name="show_suggestions")
    async def show_suggestions_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "show_suggestions", "upgrade suggestions requested")
        await ctx.reply(embed=build_suggestions_embed(), mention_author=False)

    @bot.command(name="approve_upgrade", aliases=["confirm"])
    async def approve_upgrade_command(ctx: commands.Context, password: str = "") -> None:
        if not await ensure_admin(ctx):
            return
        if not password:
            await ctx.reply(
                embed=make_embed(
                    "Password required",
                    f"Use `{config.prefix}approve_upgrade PASSWORD` or `{config.prefix}confirm PASSWORD`.",
                    discord.Color.orange(),
                ),
                mention_author=False,
            )
            return
        result = await approve_pending_upgrade(bot, config, upgrade_state, ctx, password)
        await ctx.reply(embed=result, mention_author=False)

    @bot.command(name="reject_upgrade")
    async def reject_upgrade_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        if upgrade_state.get("status") != "pending":
            await ctx.reply(embed=make_embed("No pending upgrade", "There is no upgrade waiting for approval.", discord.Color.light_grey()), mention_author=False)
            return
        upgrade_state.update(
            {
                "status": "rejected",
                "rejected_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": 0.0,
            }
        )
        log_admin_action(ctx, "reject_upgrade", "pending upgrade rejected")
        await dm_admins(bot, config, make_embed("Upgrade expired/no changes applied", "Upgrade was rejected by an admin.", discord.Color.orange()))
        await ctx.reply(embed=make_embed("Upgrade rejected", "No files were changed.", discord.Color.orange()), mention_author=False)

    @bot.command(name="hardkill")
    async def hardkill_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        if not config.allow_hardkill:
            await ctx.reply(
                embed=make_embed("Hardkill disabled", "ALLOW_DISCORD_HARDKILL is false.", discord.Color.orange()),
                mention_author=False,
            )
            return
        log_admin_action(ctx, "hardkill", "requested stop of main.py and trade_analyzer.py")
        stopped = await asyncio.to_thread(stop_trading_processes, include_analyzer=True)
        await ctx.reply(
            embed=make_embed(
                "Hardkill sent",
                "Stopped matching main.py/trade_analyzer.py processes. discord_bot.py remains online.\n"
                f"PIDs signaled: `{', '.join(str(pid) for pid in stopped) if stopped else 'none found'}`",
                discord.Color.orange(),
            ),
            mention_author=False,
        )

    @bot.command(name="restart")
    async def restart_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        log_admin_action(ctx, "restart", "requested main.py restart")
        stopped = await asyncio.to_thread(stop_trading_processes, include_analyzer=False)
        await asyncio.sleep(2)
        started = await asyncio.to_thread(start_main_bot_process)
        await ctx.reply(
            embed=make_embed(
                "Trading bot restart requested",
                f"Stopped PIDs: `{', '.join(str(pid) for pid in stopped) if stopped else 'none found'}`\n{started}",
                discord.Color.green() if started.startswith("Started") else discord.Color.orange(),
            ),
            mention_author=False,
        )

    @bot.command(name="close")
    async def close_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        if not config.allow_close_position:
            await ctx.reply(
                embed=make_embed("Close disabled", "ALLOW_DISCORD_CLOSE_POSITION is false.", discord.Color.orange()),
                mention_author=False,
            )
            return
        if not config.has_credentials:
            await ctx.reply(embed=make_embed("Close blocked", "Binance API credentials are missing.", discord.Color.red()), mention_author=False)
            return
        try:
            snapshot = await asyncio.to_thread(fetch_position_snapshot, config)
        except Exception as exc:
            await ctx.reply(embed=make_embed("Close blocked", format_binance_error(exc), discord.Color.red()), mention_author=False)
            return
        if not snapshot.is_open:
            await ctx.reply(embed=make_embed("No open position", "There is no Binance position to close.", discord.Color.light_grey()), mention_author=False)
            return
        pending_close_confirmations[ctx.author.id] = time.monotonic() + CLOSE_CONFIRM_SECONDS
        log_admin_action(ctx, "close_requested", f"side={snapshot.side} quantity={snapshot.abs_quantity}")
        embed = build_position_embed(snapshot, config.symbol, title="Confirm reduceOnly close")
        embed.add_field(
            name="Required confirmation",
            value=f"Type `{config.prefix}confirmclose` within `{CLOSE_CONFIRM_SECONDS}` seconds.",
            inline=False,
        )
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="confirmclose")
    async def confirm_close_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        deadline = pending_close_confirmations.get(ctx.author.id)
        if deadline is None or time.monotonic() > deadline:
            pending_close_confirmations.pop(ctx.author.id, None)
            await ctx.reply(
                embed=make_embed("Close confirmation expired", f"Run `{config.prefix}close` again if you still want to close.", discord.Color.orange()),
                mention_author=False,
            )
            return
        pending_close_confirmations.pop(ctx.author.id, None)
        log_admin_action(ctx, "close_confirmed", "confirmed reduceOnly market close")
        try:
            result = await asyncio.to_thread(close_position_reduce_only, config)
        except Exception as exc:
            log_admin_action(ctx, "close_failed", format_binance_error(exc))
            await ctx.reply(embed=make_embed("Close failed", format_binance_error(exc), discord.Color.red()), mention_author=False)
            return
        log_admin_action(ctx, "close_sent", json.dumps(result, default=str))
        embed = make_embed("ReduceOnly close sent", color=discord.Color.green())
        embed.add_field(name="Symbol", value=f"`{result['symbol']}`", inline=True)
        embed.add_field(name="Side Closed", value=f"`{result['position_side']}`", inline=True)
        embed.add_field(name="Quantity", value=f"`{result['quantity']}`", inline=True)
        embed.add_field(name="Estimated PnL", value=f"`{result['estimated_pnl']} USDC`", inline=True)
        embed.add_field(name="Close Order Side", value=f"`{result['order_side']}`", inline=True)
        embed.add_field(name="Reduce Only", value="`true`", inline=True)
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="pause")
    async def pause_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": str(ctx.author),
            "created_by_id": ctx.author.id,
            "reason": "Discord pause command",
        }
        PAUSED_LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log_admin_action(ctx, "pause", "PAUSED.lock created")
        await ctx.reply(
            embed=make_embed("Trading paused", "New entries are blocked. Existing position management continues.", discord.Color.orange()),
            mention_author=False,
        )

    @bot.command(name="resume")
    async def resume_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        if PAUSED_LOCK_PATH.exists():
            PAUSED_LOCK_PATH.unlink()
            detail = "PAUSED.lock removed"
        else:
            detail = "PAUSED.lock was already absent"
        log_admin_action(ctx, "resume", detail)
        await ctx.reply(
            embed=make_embed("Trading resumed", "New entries are allowed again unless another safety limit blocks them.", discord.Color.green()),
            mention_author=False,
        )

    @bot.command(name="lastsignal")
    async def last_signal_command(ctx: commands.Context) -> None:
        if not await ensure_admin(ctx):
            return
        signal_line = find_last_log_event(tail_lines(BOT_LOG_PATH, config.status_tail_lines), ("Signal=", "last signal:"))
        await ctx.reply(
            embed=make_embed("Latest strategy signal", signal_line or "No strategy signal found in bot.log.", discord.Color.blue()),
            mention_author=False,
        )

    @bot.command(name="trades")
    async def trades_command(ctx: commands.Context, count: int = 5) -> None:
        if not await ensure_admin(ctx):
            return
        count = max(1, min(count, 10))
        await ctx.reply(embed=build_recent_trades_embed(count), mention_author=False)

    @bot.command(name="logs")
    async def logs_command(ctx: commands.Context, count: int = 15) -> None:
        if not await ensure_admin(ctx):
            return
        count = max(1, min(count, 40))
        lines = tail_lines(BOT_LOG_PATH, count)
        description = "bot.log does not exist yet."
        if lines:
            description = "```text\n" + truncate("\n".join(clean_log_line(line) for line in lines), 1800) + "\n```"
        await ctx.reply(embed=make_embed(f"Last {count} log lines", description, discord.Color.dark_grey()), mention_author=False)

    return bot


def new_upgrade_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "created_at": None,
        "expires_at": 0.0,
        "safe_updates": {},
        "blocked_updates": {},
        "risk_level": "none",
        "learning_state": {},
        "last_notice_at": 0.0,
    }


async def auto_upgrade_loop(
    bot: commands.Bot,
    config: DiscordBotConfig,
    upgrade_state: dict[str, Any],
) -> None:
    if not config.auto_upgrade_enabled:
        logging.getLogger("discord_bot").info("Auto-upgrade scheduler disabled.")
        return

    interval_seconds = config.auto_upgrade_interval_hours * 60 * 60
    next_run_at = time.monotonic() + interval_seconds
    logging.getLogger("discord_bot").info(
        "Auto-upgrade scheduler enabled. interval_hours=%s",
        config.auto_upgrade_interval_hours,
    )

    while not bot.is_closed():
        try:
            await expire_pending_upgrade_if_needed(bot, config, upgrade_state)
            await apply_waiting_upgrade_if_flat(bot, config, upgrade_state)
            if (
                time.monotonic() >= next_run_at
                and upgrade_state.get("status") not in {"pending", "approved_waiting_flat", "applying"}
            ):
                await run_auto_upgrade_analysis(bot, config, upgrade_state)
                next_run_at = time.monotonic() + interval_seconds
        except Exception as exc:
            logging.getLogger("discord_bot").exception("Auto-upgrade loop error: %s", exc)
            await dm_admins(
                bot,
                config,
                make_embed("Auto-upgrade manager error", truncate(str(exc), 1500), discord.Color.red()),
            )
        await asyncio.sleep(UPGRADE_CHECK_SECONDS)


async def run_auto_upgrade_analysis(
    bot: commands.Bot,
    config: DiscordBotConfig,
    upgrade_state: dict[str, Any],
) -> None:
    logging.getLogger("discord_bot").info("Running scheduled approval-gated auto-upgrade analysis.")
    result = await run_analyzer_subprocess(discord_alert=False)
    if result["returncode"] != 0:
        await dm_admins(
            bot,
            config,
            make_embed(
                "Upgrade analysis failed",
                truncate(result.get("stderr") or result.get("stdout") or "No output.", 1600),
                discord.Color.red(),
            ),
        )
        return

    state = load_learning_state()
    safe_updates, blocked_updates = load_safe_suggested_env_updates()
    if not safe_updates:
        upgrade_state.clear()
        upgrade_state.update(new_upgrade_state())
        await dm_admins(
            bot,
            config,
            make_embed(
                "Learning analysis completed",
                "No safe `.env` upgrade is currently recommended. No changes were applied.",
                discord.Color.orange(),
            ),
        )
        return

    expires_at = time.monotonic() + UPGRADE_APPROVAL_SECONDS
    upgrade_state.clear()
    upgrade_state.update(
        {
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at,
            "safe_updates": safe_updates,
            "blocked_updates": blocked_updates,
            "risk_level": risk_level_for_updates(safe_updates),
            "learning_state": state,
            "last_notice_at": time.monotonic(),
        }
    )
    await dm_admins(bot, config, build_upgrade_ready_embed(config, upgrade_state))


async def expire_pending_upgrade_if_needed(
    bot: commands.Bot,
    config: DiscordBotConfig,
    upgrade_state: dict[str, Any],
) -> None:
    if upgrade_state.get("status") != "pending":
        return
    expires_at = float(upgrade_state.get("expires_at") or 0.0)
    if time.monotonic() <= expires_at:
        return
    upgrade_state.update(
        {
            "status": "expired",
            "expires_at": 0.0,
            "expired_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    await dm_admins(
        bot,
        config,
        make_embed("Upgrade expired/no changes applied", "Approval window expired after 10 minutes.", discord.Color.orange()),
    )


async def apply_waiting_upgrade_if_flat(
    bot: commands.Bot,
    config: DiscordBotConfig,
    upgrade_state: dict[str, Any],
) -> None:
    if upgrade_state.get("status") != "approved_waiting_flat":
        return

    try:
        snapshot = await asyncio.to_thread(fetch_position_snapshot, config)
    except Exception as exc:
        if time.monotonic() - float(upgrade_state.get("last_notice_at") or 0.0) > 600:
            upgrade_state["last_notice_at"] = time.monotonic()
            await dm_admins(
                bot,
                config,
                make_embed(
                    "Upgrade waiting",
                    f"Could not confirm position is flat yet: {format_binance_error(exc)}",
                    discord.Color.orange(),
                ),
            )
        return

    if snapshot.is_open:
        if time.monotonic() - float(upgrade_state.get("last_notice_at") or 0.0) > 600:
            upgrade_state["last_notice_at"] = time.monotonic()
            await dm_admins(
                bot,
                config,
                make_embed(
                    "Upgrade waiting for flat position",
                    f"Open {snapshot.side} position remains. New entries are paused; no changes applied yet.",
                    discord.Color.orange(),
                ),
            )
        return

    await apply_approved_upgrade(bot, config, upgrade_state, approved_by="auto-flat-check")


async def approve_pending_upgrade(
    bot: commands.Bot,
    config: DiscordBotConfig,
    upgrade_state: dict[str, Any],
    ctx: commands.Context,
    password: str,
) -> discord.Embed:
    if upgrade_state.get("status") != "pending":
        return make_embed("No pending upgrade", "There is no upgrade waiting for approval.", discord.Color.light_grey())
    if time.monotonic() > float(upgrade_state.get("expires_at") or 0.0):
        upgrade_state.update({"status": "expired", "expires_at": 0.0})
        await dm_admins(bot, config, make_embed("Upgrade expired/no changes applied", "Approval arrived too late.", discord.Color.orange()))
        return make_embed("Upgrade expired", "No changes were applied.", discord.Color.orange())
    if not config.upgrade_confirm_password:
        return make_embed("Upgrade blocked", "UPGRADE_CONFIRM_PASSWORD is missing in .env.", discord.Color.red())
    if password != config.upgrade_confirm_password:
        log_admin_action(ctx, "approve_upgrade_failed", "wrong password")
        return make_embed("Wrong password", "Upgrade was not approved.", discord.Color.red())

    log_admin_action(ctx, "approve_upgrade", "password accepted")
    create_pause_lock("Auto-approved upgrade pending")
    try:
        snapshot = await asyncio.to_thread(fetch_position_snapshot, config)
    except Exception as exc:
        upgrade_state.update({"status": "error", "error": format_binance_error(exc)})
        await dm_admins(
            bot,
            config,
            make_embed("Upgrade blocked", f"Could not confirm open position status: {format_binance_error(exc)}", discord.Color.red()),
        )
        return make_embed("Upgrade blocked", "Could not confirm position status. No changes applied.", discord.Color.red())

    if snapshot.is_open:
        upgrade_state.update(
            {
                "status": "approved_waiting_flat",
                "approved_by": ctx.author.id,
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "last_notice_at": time.monotonic(),
            }
        )
        await dm_admins(bot, config, build_waiting_for_flat_embed(snapshot))
        return make_embed(
            "Upgrade approved, waiting",
            "New entries are paused. Existing position management continues. Upgrade will apply after the position is flat.",
            discord.Color.orange(),
        )

    return await apply_approved_upgrade(bot, config, upgrade_state, approved_by=str(ctx.author.id))


async def apply_approved_upgrade(
    bot: commands.Bot,
    config: DiscordBotConfig,
    upgrade_state: dict[str, Any],
    approved_by: str,
) -> discord.Embed:
    upgrade_state.update({"status": "applying", "approved_by": approved_by})
    try:
        result = await asyncio.to_thread(apply_upgrade_sync, config, upgrade_state)
    except Exception as exc:
        upgrade_state.update({"status": "error", "error": str(exc)})
        await dm_admins(bot, config, make_embed("Upgrade failed", truncate(str(exc), 1500), discord.Color.red()))
        return make_embed("Upgrade failed", truncate(str(exc), 1500), discord.Color.red())

    upgrade_state.update(
        {
            "status": "applied",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "apply_result": result,
        }
    )
    embed = make_embed("Upgrade applied", color=discord.Color.green())
    embed.add_field(name="Applied .env changes", value=truncate(format_dict_lines(result["applied_updates"]), MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Backup", value=f"`{result['backup_dir']}`", inline=False)
    embed.add_field(name="Git", value=truncate(result["git_result"], MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Restart", value=truncate(result["restart_result"], MAX_FIELD_LENGTH), inline=False)
    await dm_admins(bot, config, embed)
    return embed


def apply_upgrade_sync(config: DiscordBotConfig, upgrade_state: dict[str, Any]) -> dict[str, Any]:
    safe_updates = dict(upgrade_state.get("safe_updates") or {})
    if not safe_updates:
        raise RuntimeError("No safe .env updates are available to apply.")

    create_pause_lock("Applying approved auto-upgrade")
    backup_dir = create_backup()
    applied_updates = apply_env_updates(safe_updates)
    git_result = commit_upgrade_changes()
    stopped = stop_trading_processes(include_analyzer=False)
    time.sleep(2)
    restart_result = start_main_bot_process()
    if PAUSED_LOCK_PATH.exists():
        PAUSED_LOCK_PATH.unlink()
    return {
        "backup_dir": str(backup_dir),
        "applied_updates": applied_updates,
        "stopped_pids": stopped,
        "git_result": git_result,
        "restart_result": restart_result,
        "auto_code_upgrade_enabled": config.auto_code_upgrade_enabled,
    }


def create_pause_lock(reason: str) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "MoneyMaker auto-upgrade",
        "reason": reason,
    }
    PAUSED_LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_backup() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUPS_DIR / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in [
        ENV_PATH,
        MAIN_BOT_PATH,
        Path(__file__).resolve(),
        TRADE_ANALYZER_PATH,
        LEARNING_REPORT_PATH,
        LEARNING_STATE_PATH,
        SUGGESTED_ENV_UPDATE_PATH,
        SUGGESTED_STRATEGY_UPDATE_PATH,
    ]:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def load_learning_state() -> dict[str, Any]:
    if not LEARNING_STATE_PATH.exists():
        return {}
    try:
        return json.loads(LEARNING_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_safe_suggested_env_updates() -> tuple[dict[str, str], dict[str, str]]:
    suggested_updates = parse_suggested_env_file()
    current_values = parse_env_file(ENV_PATH)
    safe_updates: dict[str, str] = {}
    blocked_updates: dict[str, str] = {}
    for key, value in suggested_updates.items():
        reason = blocked_update_reason(key, value, current_values)
        if reason:
            blocked_updates[key] = reason
            continue
        safe_updates[key] = value
    return safe_updates, blocked_updates


def parse_suggested_env_file() -> dict[str, str]:
    if not SUGGESTED_ENV_UPDATE_PATH.exists():
        return {}
    updates: dict[str, str] = {}
    for raw_line in SUGGESTED_ENV_UPDATE_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if key:
            updates[key] = value
    return updates


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()
    return values


def blocked_update_reason(key: str, value: str, current_values: dict[str, str]) -> Optional[str]:
    key = key.upper()
    if key in SIZE_ENV_KEYS:
        return "size/leverage updates are blocked"
    if any(fragment in key for fragment in PROTECTED_ENV_KEY_FRAGMENTS):
        return "protected credential/control key"
    if key not in SAFE_ENV_UPGRADE_KEYS:
        return "not in safe upgrade allowlist"
    if not stricter_or_new_filter_value(key, value, current_values.get(key)):
        return "not stricter than current value"
    return None


def stricter_or_new_filter_value(key: str, proposed: str, current: Optional[str]) -> bool:
    if current is None or not current.strip():
        return True
    proposed_bool = parse_bool_safely(proposed)
    current_bool = parse_bool_safely(current)
    if proposed_bool is not None or current_bool is not None:
        return proposed_bool is True and current_bool is not True

    proposed_decimal = parse_decimal(proposed)
    current_decimal = parse_decimal(current)
    if proposed_decimal is None or current_decimal is None:
        return proposed.strip() != current.strip()
    if key.startswith("MAX_"):
        return proposed_decimal <= current_decimal
    return proposed_decimal >= current_decimal


def parse_bool_safely(value: str) -> Optional[bool]:
    try:
        return parse_bool(value, default=False)
    except RuntimeError:
        return None


def apply_env_updates(updates: dict[str, str]) -> dict[str, str]:
    if not ENV_PATH.exists():
        raise RuntimeError(".env file is missing.")
    original_lines = ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    remaining = {key.upper(): value for key, value in updates.items()}
    output_lines: list[str] = []
    applied: dict[str, str] = {}

    for line in original_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output_lines.append(line)
            continue
        key, _old_value = line.split("=", 1)
        normalized_key = key.strip().upper()
        if normalized_key in remaining:
            value = remaining.pop(normalized_key)
            output_lines.append(f"{normalized_key}={value}")
            applied[normalized_key] = value
        else:
            output_lines.append(line)

    if remaining:
        output_lines.extend(["", "# Auto-approved safe strategy filter updates"])
        for key, value in remaining.items():
            output_lines.append(f"{key}={value}")
            applied[key] = value

    ENV_PATH.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
    return applied


def commit_upgrade_changes() -> str:
    git = find_executable("git")
    if not git:
        return "git not found; skipped local commit"

    tracked_env = subprocess.run(
        [git, "ls-files", "--error-unmatch", ".env"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    paths = [
        "learning_report.md",
        "learning_state.json",
        "suggested_env_update.txt",
        "suggested_strategy_update.md",
    ]
    if tracked_env.returncode == 0:
        paths.append(".env")

    add_result = subprocess.run([git, "add", *paths], cwd=PROJECT_DIR, capture_output=True, text=True, check=False)
    if add_result.returncode != 0:
        return f"git add failed: {truncate(add_result.stderr or add_result.stdout, 800)}"

    diff_result = subprocess.run([git, "diff", "--cached", "--quiet"], cwd=PROJECT_DIR, check=False)
    if diff_result.returncode == 0:
        return "no tracked changes to commit"

    commit_result = subprocess.run(
        [git, "commit", "-m", "Auto-approved strategy update"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_result.returncode != 0:
        return f"git commit failed: {truncate(commit_result.stderr or commit_result.stdout, 800)}"
    return truncate(commit_result.stdout.strip() or "local commit created", 1000)


def build_upgrade_ready_embed(config: DiscordBotConfig, upgrade_state: dict[str, Any]) -> discord.Embed:
    state = upgrade_state.get("learning_state", {})
    metrics = state.get("metrics", {})
    patterns = state.get("detected_losing_patterns", [])
    top_pattern = patterns[0].get("pattern", "none detected") if patterns else "none detected"
    embed = make_embed(
        "Upgrade suggestion ready",
        f"Reply `{config.prefix}confirm PASSWORD` or `{config.prefix}approve_upgrade PASSWORD` within 10 minutes.",
        discord.Color.gold(),
    )
    embed.add_field(name="Win rate", value=f"`{metrics.get('win_rate_pct', 0)}%`", inline=True)
    embed.add_field(name="Profit factor", value=f"`{metrics.get('profit_factor', 0)}`", inline=True)
    embed.add_field(name="Risk level", value=f"`{upgrade_state.get('risk_level', 'unknown')}`", inline=True)
    embed.add_field(name="Top losing pattern", value=truncate(top_pattern, MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Proposed .env changes", value=truncate(format_dict_lines(upgrade_state.get("safe_updates", {})), MAX_FIELD_LENGTH), inline=False)
    blocked = upgrade_state.get("blocked_updates") or {}
    if blocked:
        embed.add_field(name="Blocked suggestions", value=truncate(format_dict_lines(blocked), MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Code changes", value="`not auto-applied` unless AUTO_CODE_UPGRADE_ENABLED=true", inline=False)
    return embed


def build_waiting_for_flat_embed(snapshot: PositionSnapshot) -> discord.Embed:
    embed = make_embed(
        "Upgrade approved, waiting for flat position",
        "New entries are paused. Existing SL/TP management continues. No files changed yet.",
        discord.Color.orange(),
    )
    embed.add_field(name="Open side", value=f"`{snapshot.side}`", inline=True)
    embed.add_field(name="Quantity", value=f"`{format_decimal(snapshot.abs_quantity)}`", inline=True)
    embed.add_field(name="Entry", value=f"`{format_decimal(snapshot.entry_price)}`", inline=True)
    return embed


def build_upgrade_status_embed(upgrade_state: dict[str, Any], config: DiscordBotConfig) -> discord.Embed:
    embed = make_embed("Auto-upgrade status", color=discord.Color.blue())
    embed.add_field(name="Enabled", value=f"`{str(config.auto_upgrade_enabled).lower()}`", inline=True)
    embed.add_field(name="Interval", value=f"`{config.auto_upgrade_interval_hours}h`", inline=True)
    embed.add_field(name="Status", value=f"`{upgrade_state.get('status', 'idle')}`", inline=True)
    embed.add_field(name="Created", value=f"`{upgrade_state.get('created_at') or 'n/a'}`", inline=False)
    expires_at = float(upgrade_state.get("expires_at") or 0.0)
    remaining = max(0, int(expires_at - time.monotonic())) if expires_at else 0
    embed.add_field(name="Approval expires in", value=f"`{remaining}s`", inline=True)
    embed.add_field(name="Risk level", value=f"`{upgrade_state.get('risk_level', 'none')}`", inline=True)
    embed.add_field(name="Safe updates", value=truncate(format_dict_lines(upgrade_state.get("safe_updates", {})), MAX_FIELD_LENGTH), inline=False)
    return embed


def build_suggestions_embed() -> discord.Embed:
    safe_updates, blocked_updates = load_safe_suggested_env_updates()
    embed = make_embed("Current upgrade suggestions", color=discord.Color.teal() if safe_updates else discord.Color.orange())
    embed.add_field(name="Safe .env changes", value=truncate(format_dict_lines(safe_updates), MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Blocked suggestions", value=truncate(format_dict_lines(blocked_updates), MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Suggested env file", value=file_status(SUGGESTED_ENV_UPDATE_PATH), inline=True)
    embed.add_field(name="Strategy notes", value=file_status(SUGGESTED_STRATEGY_UPDATE_PATH), inline=True)
    return embed


async def dm_admins(bot: commands.Bot, config: DiscordBotConfig, embed: discord.Embed) -> None:
    if not config.allowed_user_ids:
        return
    for user_id in config.allowed_user_ids:
        try:
            user = bot.get_user(user_id) or await bot.fetch_user(user_id)
            await user.send(embed=embed)
        except Exception as exc:
            logging.getLogger("discord_bot").warning("Could not DM admin user_id=%s: %s", user_id, exc)


def risk_level_for_updates(updates: dict[str, str]) -> str:
    if not updates:
        return "none"
    if len(updates) <= 2:
        return "low"
    if len(updates) <= 5:
        return "medium"
    return "high-review"


def format_dict_lines(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return "\n".join(f"{key}={value}" for key, value in values.items())


def build_binance_client(config: DiscordBotConfig) -> UMFutures:
    kwargs: dict[str, Any] = {
        "key": config.api_key,
        "secret": config.api_secret,
        "timeout": config.request_timeout,
        "base_url": config.futures_base_url,
    }
    return UMFutures(**kwargs)


def fetch_position_snapshot(config: DiscordBotConfig) -> PositionSnapshot:
    if not config.has_credentials:
        return PositionSnapshot(
            side="FLAT",
            quantity=Decimal("0"),
            entry_price=Decimal("0"),
            mark_price=Decimal("0"),
            unrealized_pnl=Decimal("0"),
        )
    client = build_binance_client(config)
    positions = client.get_position_risk(symbol=config.symbol)
    mark_price = get_mark_price(client, config.symbol)
    for item in positions:
        quantity = Decimal(str(item.get("positionAmt", "0")))
        if quantity != 0:
            side = "LONG" if quantity > 0 else "SHORT"
            return PositionSnapshot(
                side=side,
                quantity=quantity,
                entry_price=Decimal(str(item.get("entryPrice", "0"))),
                mark_price=mark_price,
                unrealized_pnl=Decimal(str(item.get("unRealizedProfit", "0"))),
            )
    return PositionSnapshot(
        side="FLAT",
        quantity=Decimal("0"),
        entry_price=Decimal("0"),
        mark_price=mark_price,
        unrealized_pnl=Decimal("0"),
    )


def close_position_reduce_only(config: DiscordBotConfig) -> dict[str, str]:
    client = build_binance_client(config)
    snapshot = fetch_position_snapshot(config)
    if not snapshot.is_open:
        raise RuntimeError("No open Binance position found.")

    close_side = "SELL" if snapshot.quantity > 0 else "BUY"
    quantity = snapshot.abs_quantity
    estimated_pnl = calculate_position_pnl(snapshot)
    order = client.new_order(
        symbol=config.symbol,
        side=close_side,
        type="MARKET",
        quantity=format_decimal(quantity),
        reduceOnly="true",
    )
    return {
        "symbol": config.symbol,
        "position_side": snapshot.side,
        "quantity": format_decimal(quantity),
        "estimated_pnl": format_decimal(estimated_pnl),
        "order_side": close_side,
        "order_id": str(order.get("orderId", "unknown")),
    }


def get_mark_price(client: UMFutures, symbol: str) -> Decimal:
    payload = client.mark_price(symbol=symbol)
    return Decimal(str(payload.get("markPrice", "0")))


def calculate_position_pnl(snapshot: PositionSnapshot) -> Decimal:
    if not snapshot.is_open or snapshot.entry_price <= 0:
        return Decimal("0")
    if snapshot.side == "LONG":
        return (snapshot.mark_price - snapshot.entry_price) * snapshot.abs_quantity
    if snapshot.side == "SHORT":
        return (snapshot.entry_price - snapshot.mark_price) * snapshot.abs_quantity
    return Decimal("0")


def build_position_embed(
    snapshot: PositionSnapshot,
    symbol: str,
    title: str = "Open Binance position",
) -> discord.Embed:
    color = discord.Color.green() if snapshot.is_open else discord.Color.light_grey()
    embed = make_embed(title, color=color)
    embed.add_field(name="Symbol", value=f"`{symbol}`", inline=True)
    embed.add_field(name="Side", value=f"`{snapshot.side}`", inline=True)
    embed.add_field(name="Quantity", value=f"`{format_decimal(snapshot.abs_quantity)}`", inline=True)
    embed.add_field(name="Entry", value=f"`{format_decimal(snapshot.entry_price)}`", inline=True)
    embed.add_field(name="Mark", value=f"`{format_decimal(snapshot.mark_price)}`", inline=True)
    embed.add_field(name="Unrealized PnL", value=f"`{format_decimal(snapshot.unrealized_pnl)} USDC`", inline=True)
    return embed


def build_pnl_embed() -> discord.Embed:
    rows = load_trade_rows()
    if not rows:
        return make_embed("PnL summary", "No trades.csv rows found yet.", discord.Color.light_grey())

    today = datetime.now(timezone.utc).date()
    total_pnl = Decimal("0")
    today_pnl = Decimal("0")
    wins = 0
    losses = 0
    today_trades = 0

    for row in rows:
        pnl = parse_decimal(row.get("pnl_usdt"))
        if pnl is None:
            continue
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        timestamp = parse_timestamp(row.get("timestamp"))
        if timestamp and timestamp.date() == today:
            today_pnl += pnl
            today_trades += 1

    total_count = len(rows)
    win_rate = (Decimal(wins) / Decimal(total_count) * Decimal("100")) if total_count else Decimal("0")
    embed = make_embed("PnL summary", color=discord.Color.green() if total_pnl >= 0 else discord.Color.red())
    embed.add_field(name="Total PnL", value=f"`{format_decimal(total_pnl)} USDC`", inline=True)
    embed.add_field(name="UTC Today PnL", value=f"`{format_decimal(today_pnl)} USDC`", inline=True)
    embed.add_field(name="Total trades", value=f"`{total_count}`", inline=True)
    embed.add_field(name="Today trades", value=f"`{today_trades}`", inline=True)
    embed.add_field(name="Wins / Losses", value=f"`{wins} / {losses}`", inline=True)
    embed.add_field(name="Win rate", value=f"`{format_decimal(win_rate)}%`", inline=True)
    return embed


def build_recent_trades_embed(count: int) -> discord.Embed:
    rows = load_trade_rows()
    if not rows:
        return make_embed("Recent trades", "No trades.csv rows found yet.", discord.Color.light_grey())

    embed = make_embed(f"Last {min(count, len(rows))} closed trades", color=discord.Color.gold())
    for row in rows[-count:][::-1]:
        timestamp = row.get("timestamp", "unknown")
        side = row.get("side", "unknown")
        pnl = row.get("pnl_usdt", "n/a")
        exit_reason = row.get("exit_reason", "n/a")
        value = (
            f"Side: `{side}`\n"
            f"Qty: `{row.get('quantity', 'n/a')}`\n"
            f"Entry: `{row.get('entry_price', 'n/a')}` -> Exit: `{row.get('exit_price', 'n/a')}`\n"
            f"PnL: `{pnl}`\n"
            f"Exit: {truncate(exit_reason, 250)}"
        )
        embed.add_field(name=truncate(timestamp, 90), value=truncate(value, MAX_FIELD_LENGTH), inline=False)
    return embed


def build_report_embed() -> discord.Embed:
    if not LEARNING_REPORT_PATH.exists():
        return make_embed("Learning report", "learning_report.md does not exist yet.", discord.Color.light_grey())
    report = LEARNING_REPORT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    report = truncate(report, MAX_DESCRIPTION_LENGTH)
    embed = make_embed("Learning report summary", report or "learning_report.md is empty.", discord.Color.green())
    embed.add_field(name="learning_state.json", value=file_status(LEARNING_STATE_PATH), inline=True)
    embed.add_field(name="suggested_env_update.txt", value=file_status(SUGGESTED_ENV_UPDATE_PATH), inline=True)
    embed.add_field(name="suggested_strategy_update.md", value=file_status(SUGGESTED_STRATEGY_UPDATE_PATH), inline=True)
    return embed


def build_learning_state_embed(title: str) -> discord.Embed:
    if not LEARNING_STATE_PATH.exists():
        return make_embed(title, "learning_state.json was not created.", discord.Color.orange())
    try:
        state = json.loads(LEARNING_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return make_embed(title, f"Could not read learning_state.json: {exc}", discord.Color.red())

    metrics = state.get("metrics", {})
    recommendations = state.get("recommended_parameter_changes", [])
    patterns = state.get("detected_losing_patterns", [])
    total_pnl = safe_decimal(metrics.get("gross_profit")) - safe_decimal(metrics.get("gross_loss"))
    embed = make_embed(title, color=discord.Color.teal() if recommendations else discord.Color.orange())
    embed.add_field(name="Total trades", value=f"`{metrics.get('total_trades', 0)}`", inline=True)
    embed.add_field(name="Win rate", value=f"`{metrics.get('win_rate_pct', 0)}%`", inline=True)
    embed.add_field(name="Profit factor", value=f"`{metrics.get('profit_factor', 0)}`", inline=True)
    embed.add_field(name="Total PnL", value=f"`{format_decimal(total_pnl)} USDC`", inline=True)
    embed.add_field(name="Suggestions", value="`yes`" if recommendations else "`no`", inline=True)
    embed.add_field(name="Suggested only", value=f"`{str(state.get('suggested_only', True)).lower()}`", inline=True)
    embed.add_field(name="Top losing patterns", value=truncate(format_patterns(patterns), MAX_FIELD_LENGTH), inline=False)
    embed.add_field(name="Suggested stricter filters", value=truncate(format_recommendations(recommendations), MAX_FIELD_LENGTH), inline=False)
    if recommendations:
        embed.add_field(name="Review required", value="Env update suggested. Review suggested_env_update.txt", inline=False)
    return embed


async def run_analyzer_subprocess(discord_alert: bool = True) -> dict[str, Any]:
    if not TRADE_ANALYZER_PATH.exists():
        return {"returncode": 1, "stdout": "", "stderr": "trade_analyzer.py was not found."}
    env = os.environ.copy()
    env["ANALYSIS_DISCORD_ALERT"] = "true" if discord_alert else "false"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(TRADE_ANALYZER_PATH),
        cwd=PROJECT_DIR,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return {"returncode": 124, "stdout": "", "stderr": "trade_analyzer.py timed out."}
    return {
        "returncode": process.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


def stop_trading_processes(include_analyzer: bool) -> list[int]:
    patterns = ["main.py"]
    if include_analyzer:
        patterns.append("trade_analyzer.py")
    stopped: list[int] = []
    for pid in find_matching_pids(patterns):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except OSError:
            continue
    return stopped


def find_matching_pids(patterns: list[str]) -> list[int]:
    if os.name == "nt":
        return []
    pids: list[int] = []
    proc_dir = Path("/proc")
    if proc_dir.exists():
        for child in proc_dir.iterdir():
            if not child.name.isdigit():
                continue
            pid = int(child.name)
            command_line = read_process_command_line(pid)
            if not command_line or "discord_bot.py" in command_line:
                continue
            if any(pattern in command_line for pattern in patterns):
                pids.append(pid)
        return sorted(set(pids))

    for pattern in patterns:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        for line in result.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)
    return sorted(set(pids))


def read_process_command_line(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="replace")


def start_main_bot_process() -> str:
    if not MAIN_BOT_PATH.exists():
        return "main.py was not found."
    screen = find_executable("screen")
    if screen:
        result = subprocess.run(
            [screen, "-dmS", TRADE_SCREEN_NAME, sys.executable, str(MAIN_BOT_PATH)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return f"Started main.py in screen session `{TRADE_SCREEN_NAME}`."
        return f"screen start failed: {truncate(result.stderr or result.stdout, 800)}"

    with (PROJECT_DIR / "trade_bot.out.log").open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            [sys.executable, str(MAIN_BOT_PATH)],
            cwd=PROJECT_DIR,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
        )
    return "Started main.py with subprocess fallback."


def find_executable(name: str) -> Optional[str]:
    for directory in os.getenv("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def log_admin_action(ctx: commands.Context, action: str, detail: str) -> None:
    line = (
        f"{datetime.now(timezone.utc).isoformat()} | "
        f"user={ctx.author} | user_id={ctx.author.id} | channel_id={ctx.channel.id} | "
        f"action={action} | detail={detail}\n"
    )
    try:
        with ADMIN_ACTION_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        logging.getLogger("discord_bot").warning("Could not write admin action log: %s", exc)


def load_trade_rows() -> list[dict[str, str]]:
    if not TRADES_CSV_PATH.exists():
        return []
    with TRADES_CSV_PATH.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def find_last_log_event(lines: Iterable[str], needles: tuple[str, ...]) -> Optional[str]:
    for line in reversed(list(lines)):
        if any(needle in line for needle in needles):
            return truncate(clean_log_line(line), MAX_FIELD_LENGTH)
    return None


def clean_log_line(line: str) -> str:
    parts = line.split(" | ", 3)
    if len(parts) == 4:
        return parts[3].strip()
    return line.strip()


def file_status(path: Path) -> str:
    if not path.exists():
        return "`missing`"
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return f"`exists` modified `{modified.isoformat(timespec='seconds')}`"


def make_embed(
    title: str,
    description: str = "",
    color: discord.Color = discord.Color.blurple(),
) -> discord.Embed:
    embed = discord.Embed(
        title=title,
        description=truncate(description, MAX_DESCRIPTION_LENGTH) if description else None,
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    embed.set_author(name="MoneyMaker", icon_url="https://cryptologos.cc/logos/xrp-xrp-logo.png")
    embed.set_footer(text="Admin-only VPS control | no leverage/order-size mutation commands")
    return embed


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None or not value.strip():
        return None
    try:
        return Decimal(value.strip())
    except InvalidOperation:
        return None


def safe_decimal(value: Any) -> Decimal:
    try:
        if value == "inf":
            return Decimal("0")
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal("0")


def format_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") or "0"


def parse_optional_int(value: Optional[str]) -> Optional[int]:
    cleaned = normalize_optional_value(value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value: {cleaned}") from exc


def parse_positive_int(value: Optional[str], default: int) -> int:
    parsed = parse_optional_int(value)
    if parsed is None:
        return default
    return max(1, parsed)


def parse_id_set(value: Optional[str]) -> set[int]:
    cleaned = normalize_optional_value(value)
    if not cleaned:
        return set()
    ids: set[int] = set()
    for part in re.split(r"[,\s]+", cleaned):
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError as exc:
            raise RuntimeError(f"Invalid Discord user ID: {part}") from exc
    return ids


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value: {value}")


def normalize_optional_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def normalize_secret(value: Optional[str]) -> Optional[str]:
    cleaned = normalize_optional_value(value)
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if "your_" in lowered or "replace" in lowered or "placeholder" in lowered:
        return None
    return cleaned


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def format_patterns(patterns: list[dict[str, Any]]) -> str:
    if not patterns:
        return "none detected"
    return "\n".join(
        f"- {item.get('pattern', 'pattern')}: {item.get('loss_count', 0)} loss(es)"
        for item in patterns[:3]
    )


def format_recommendations(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "none"
    lines = []
    for item in recommendations[:5]:
        parameter = item.get("parameter", "filter")
        value = item.get("suggested_min", item.get("suggested_max", "manual review"))
        reason = item.get("reason", "review")
        lines.append(f"- {parameter}: {value} ({reason})")
    return "\n".join(lines)


def format_binance_error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        code = getattr(exc, "error_code", None) or getattr(exc, "code", "unknown")
        message = getattr(exc, "error_message", None) or getattr(exc, "message", None) or str(exc)
        return f"Binance error code={code} message={message}"
    if isinstance(exc, ServerError):
        return f"Binance server error: {exc}"
    return str(exc)


def main() -> int:
    configure_logging()
    config = load_config()
    if not config.allowed_user_ids:
        logging.getLogger("discord_bot").warning(
            "No ALLOWED_DISCORD_USER_IDS configured. Commands will be blocked."
        )
    bot = build_bot(config)
    bot.run(config.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
