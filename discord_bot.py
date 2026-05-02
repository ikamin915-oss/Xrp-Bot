#!/usr/bin/env python3
"""MoneyMaker read-only Discord command bot for the Binance Futures project.

Run this beside main.py on the same VPS. It reads local files only:
bot.log, trades.csv, learning_report.md, and learning_state.json.

It intentionally does not place, close, modify, or cancel trades.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
BOT_LOG_PATH = PROJECT_DIR / "bot.log"
TRADES_CSV_PATH = PROJECT_DIR / "trades.csv"
LEARNING_REPORT_PATH = PROJECT_DIR / "learning_report.md"
LEARNING_STATE_PATH = PROJECT_DIR / "learning_state.json"
MAX_FIELD_LENGTH = 1024
MAX_DESCRIPTION_LENGTH = 3900
DEFAULT_TAIL_LINES = 80


@dataclass(frozen=True)
class DiscordBotConfig:
    token: str
    prefix: str
    allowed_channel_id: Optional[int]
    allowed_user_ids: set[int]
    status_tail_lines: int


def load_config() -> DiscordBotConfig:
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    token = normalize_optional_value(os.getenv("DISCORD_BOT_TOKEN"))
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing in .env.")

    prefix = normalize_optional_value(os.getenv("DISCORD_COMMAND_PREFIX")) or "!"
    return DiscordBotConfig(
        token=token,
        prefix=prefix,
        allowed_channel_id=parse_optional_int(os.getenv("DISCORD_ALLOWED_CHANNEL_ID")),
        allowed_user_ids=parse_id_set(os.getenv("DISCORD_ALLOWED_USER_IDS")),
        status_tail_lines=parse_positive_int(os.getenv("DISCORD_STATUS_TAIL_LINES"), DEFAULT_TAIL_LINES),
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

    async def ensure_allowed(ctx: commands.Context) -> bool:
        if config.allowed_channel_id and ctx.channel.id != config.allowed_channel_id:
            return False
        if config.allowed_user_ids and ctx.author.id not in config.allowed_user_ids:
            await ctx.reply("This Discord user is not allowed to query the trading bot.", mention_author=False)
            return False
        return True

    @bot.event
    async def on_ready() -> None:
        logging.getLogger("discord_bot").info(
            "MoneyMaker Discord bot online as %s. prefix=%s",
            bot.user,
            config.prefix,
        )

    @bot.command(name="help")
    async def help_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        embed = make_embed(
            "MoneyMaker read-only commands",
            (
                f"`{config.prefix}status` - latest signal, position, warnings, and file health\n"
                f"`{config.prefix}position` - latest position snapshot from bot.log\n"
                f"`{config.prefix}lastsignal` - latest strategy decision\n"
                f"`{config.prefix}pnl` - total and UTC-today PnL from trades.csv\n"
                f"`{config.prefix}trades [count]` - recent closed trades\n"
                f"`{config.prefix}report` - learning_report.md summary\n"
                f"`{config.prefix}logs [count]` - recent bot.log lines\n"
                f"`{config.prefix}alive` - confirm this Discord bot is online"
            ),
            discord.Color.blurple(),
        )
        embed.set_footer(text="Read-only: this bot cannot place or close trades.")
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="alive")
    async def alive_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        await ctx.reply(embed=make_embed("MoneyMaker is online", "Read-only monitoring is active."), mention_author=False)

    @bot.command(name="status")
    async def status_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        lines = tail_lines(BOT_LOG_PATH, config.status_tail_lines)
        embed = make_embed("Trading bot status", color=discord.Color.blue())
        embed.add_field(name="Project", value=f"`{PROJECT_DIR}`", inline=False)
        embed.add_field(name="bot.log", value=file_status(BOT_LOG_PATH), inline=True)
        embed.add_field(name="trades.csv", value=file_status(TRADES_CSV_PATH), inline=True)
        embed.add_field(name="Last signal", value=find_last_log_event(lines, ("Signal=",)) or "No signal found.", inline=False)
        embed.add_field(name="Position", value=find_last_log_event(lines, ("Current position:",)) or "No position line found.", inline=False)
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
        if not await ensure_allowed(ctx):
            return
        lines = tail_lines(BOT_LOG_PATH, config.status_tail_lines)
        position = find_last_log_event(lines, ("Current position:",)) or "No position snapshot found in bot.log."
        protection = find_last_log_event(lines, ("Software-managed levels:", "software-managed SL/TP active"))
        embed = make_embed("Latest position snapshot", color=discord.Color.teal())
        embed.add_field(name="Position", value=position, inline=False)
        if protection:
            embed.add_field(name="Protection", value=protection, inline=False)
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="lastsignal")
    async def last_signal_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        signal = find_last_log_event(tail_lines(BOT_LOG_PATH, config.status_tail_lines), ("Signal=",))
        await ctx.reply(
            embed=make_embed("Latest strategy signal", signal or "No strategy signal found in bot.log.", discord.Color.blue()),
            mention_author=False,
        )

    @bot.command(name="pnl")
    async def pnl_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        embed = build_pnl_embed()
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="trades")
    async def trades_command(ctx: commands.Context, count: int = 5) -> None:
        if not await ensure_allowed(ctx):
            return
        count = max(1, min(count, 10))
        embed = build_recent_trades_embed(count)
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="report")
    async def report_command(ctx: commands.Context) -> None:
        if not await ensure_allowed(ctx):
            return
        if not LEARNING_REPORT_PATH.exists():
            await ctx.reply(embed=make_embed("Learning report", "learning_report.md does not exist yet."), mention_author=False)
            return
        report = LEARNING_REPORT_PATH.read_text(encoding="utf-8", errors="replace").strip()
        report = truncate(report, MAX_DESCRIPTION_LENGTH)
        embed = make_embed("Learning report summary", report or "learning_report.md is empty.", discord.Color.green())
        if LEARNING_STATE_PATH.exists():
            embed.add_field(name="learning_state.json", value=file_status(LEARNING_STATE_PATH), inline=False)
        await ctx.reply(embed=embed, mention_author=False)

    @bot.command(name="logs")
    async def logs_command(ctx: commands.Context, count: int = 15) -> None:
        if not await ensure_allowed(ctx):
            return
        count = max(1, min(count, 40))
        lines = tail_lines(BOT_LOG_PATH, count)
        description = "bot.log does not exist yet."
        if lines:
            description = "```text\n" + truncate("\n".join(clean_log_line(line) for line in lines), 1800) + "\n```"
        await ctx.reply(embed=make_embed(f"Last {count} log lines", description, discord.Color.dark_grey()), mention_author=False)

    return bot


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
    embed.set_footer(text="Read-only VPS monitor. Trading commands are disabled.")
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


def normalize_optional_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def main() -> int:
    configure_logging()
    config = load_config()
    bot = build_bot(config)
    bot.run(config.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
