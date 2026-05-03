#!/usr/bin/env python3
"""Suggestion-only trade review for the XRPUSDC scalping bot.

This module reads trades.csv and bot logs, writes learning_report.md, and writes
learning_state.json. It never changes live trading settings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
TRADES_CSV_PATH = PROJECT_DIR / "trades.csv"
REPORT_PATH = PROJECT_DIR / "learning_report.md"
STATE_PATH = PROJECT_DIR / "learning_state.json"
SUGGESTED_ENV_UPDATE_PATH = PROJECT_DIR / "suggested_env_update.txt"
SUGGESTED_STRATEGY_UPDATE_PATH = PROJECT_DIR / "suggested_strategy_update.md"
BOT_LOG_PATHS = [
    PROJECT_DIR / "bot.log",
    PROJECT_DIR / "main.log",
    PROJECT_DIR / "logs" / "bot.log",
]
DISCORD_TIMEOUT_SECONDS = 10
DISCORD_LEARNING_COLOR = 0x1ABC9C
DISCORD_WARNING_COLOR = 0xF1C40F
DISCORD_MAX_FIELD_VALUE_LENGTH = 1024

NUMERIC_COLUMNS = [
    "entry_price",
    "exit_price",
    "quantity",
    "pnl_usdt",
    "ema7",
    "ema25",
    "ema99",
    "ema_spread_pct",
    "candle_body_ratio",
    "distance_from_ema7_pct",
    "volume",
    "holding_time_seconds",
]


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def round_value(value: Any, places: int = 4) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), places)


def load_trades() -> pd.DataFrame:
    if not TRADES_CSV_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(TRADES_CSV_PATH)
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if "timestamp" not in frame.columns:
        frame["timestamp"] = ""
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for column in [
        "trade_id",
        "symbol",
        "side",
        "exit_reason",
        "entry_reason",
        "previous_candle_direction",
        "cooldown_status",
        "tp1_hit_before_exit",
    ]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    return enrich_trades(frame)


def enrich_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["time_window"] = frame["hour_utc"].map(lambda hour: f"{hour:02d}:00-{(hour + 1) % 24:02d}:00 UTC")
    frame["is_win"] = frame["pnl_usdt"] > 0
    frame["is_loss"] = frame["pnl_usdt"] < 0
    frame["result"] = frame["is_win"].map({True: "win", False: "loss_or_flat"})
    frame.loc[frame["pnl_usdt"] == 0, "result"] = "flat"
    frame["ema_spread_abs_pct"] = frame["ema_spread_pct"].abs()
    frame["previous_trade_was_loss"] = frame["pnl_usdt"].shift(1).fillna(0) < 0
    frame["ema_spread_bucket"] = pd.cut(
        frame["ema_spread_abs_pct"],
        bins=[-0.000001, 0.03, 0.06, 0.10, float("inf")],
        labels=["very tight <=0.03%", "tight 0.03-0.06%", "moderate 0.06-0.10%", "wide >0.10%"],
    ).astype(str)
    frame["body_ratio_bucket"] = pd.cut(
        frame["candle_body_ratio"],
        bins=[-0.000001, 0.25, 0.45, 0.65, float("inf")],
        labels=["weak <=0.25", "soft 0.25-0.45", "solid 0.45-0.65", "strong >0.65"],
    ).astype(str)
    frame["distance_ema7_bucket"] = pd.cut(
        frame["distance_from_ema7_pct"],
        bins=[-0.000001, 0.05, 0.15, 0.30, float("inf")],
        labels=["near <=0.05%", "ok 0.05-0.15%", "extended 0.15-0.30%", "far >0.30%"],
    ).astype(str)
    frame["holding_time_bucket"] = pd.cut(
        frame["holding_time_seconds"],
        bins=[-0.000001, 300, 900, 1800, 3600, float("inf")],
        labels=["<=5m", "5-15m", "15-30m", "30-60m", ">60m"],
    ).astype(str)
    frame["pnl_bucket"] = pd.cut(
        frame["pnl_usdt"],
        bins=[float("-inf"), -1.0, -0.25, 0, 0.25, 1.0, float("inf")],
        labels=["large loss", "loss", "small loss", "small win", "win", "large win"],
    ).astype(str)
    return frame


def calculate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    wins = frame[frame["pnl_usdt"] > 0]
    losses = frame[frame["pnl_usdt"] < 0]
    gross_profit = float(wins["pnl_usdt"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl_usdt"].sum())) if not losses.empty else 0.0
    profit_factor = 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    return {
        "total_trades": total,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round_value((len(wins) / total) * 100 if total else 0),
        "average_win": round_value(wins["pnl_usdt"].mean() if not wins.empty else 0),
        "average_loss": round_value(losses["pnl_usdt"].mean() if not losses.empty else 0),
        "gross_profit": round_value(gross_profit),
        "gross_loss": round_value(gross_loss),
        "profit_factor": "inf" if profit_factor == float("inf") else round_value(profit_factor),
        "max_consecutive_losses": max_consecutive_losses(frame),
    }


def max_consecutive_losses(frame: pd.DataFrame) -> int:
    longest = 0
    current = 0
    for pnl in frame["pnl_usdt"].tolist():
        if pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_group(frame: pd.DataFrame, column: str, limit: int = 8) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            trades=("pnl_usdt", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            total_pnl=("pnl_usdt", "sum"),
            average_pnl=("pnl_usdt", "mean"),
        )
        .reset_index()
    )
    grouped["win_rate_pct"] = grouped.apply(
        lambda row: (row["wins"] / row["trades"]) * 100 if row["trades"] else 0,
        axis=1,
    )
    grouped = grouped.sort_values(["total_pnl", "trades"], ascending=[False, False]).head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                str(column): str(row[column]),
                "trades": int(row["trades"]),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "win_rate_pct": round_value(row["win_rate_pct"]),
                "total_pnl": round_value(row["total_pnl"]),
                "average_pnl": round_value(row["average_pnl"]),
            }
        )
    return rows


def compare_winners_losers(frame: pd.DataFrame) -> dict[str, Any]:
    winners = frame[frame["pnl_usdt"] > 0]
    losses = frame[frame["pnl_usdt"] < 0]
    fields = ["ema_spread_abs_pct", "candle_body_ratio", "distance_from_ema7_pct", "volume", "holding_time_seconds"]
    comparison: dict[str, Any] = {}
    for field in fields:
        comparison[field] = {
            "winner_median": round_value(winners[field].median() if not winners.empty else 0),
            "loser_median": round_value(losses[field].median() if not losses.empty else 0),
        }
    return comparison


def identify_losing_patterns(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if frame.empty:
        return [], {}
    losses = frame[frame["pnl_usdt"] < 0]
    if losses.empty:
        return [], {}

    spread_cutoff = max(0.03, float(frame["ema_spread_abs_pct"].quantile(0.25)))
    weak_body_cutoff = 0.35
    far_ema_cutoff = max(0.20, float(frame["distance_from_ema7_pct"].quantile(0.75)))
    high_body_cutoff = max(0.65, float(frame["candle_body_ratio"].quantile(0.75)))
    low_volume_cutoff = float(frame["volume"].quantile(0.25))

    thresholds = {
        "tight_ema_spread_pct": round_value(spread_cutoff),
        "weak_body_ratio": round_value(weak_body_cutoff),
        "far_from_ema7_pct": round_value(far_ema_cutoff),
        "big_body_ratio": round_value(high_body_cutoff),
        "low_volume": round_value(low_volume_cutoff),
    }

    pattern_specs = [
        (
            "losses after EMA spread too tight",
            losses["ema_spread_abs_pct"] <= spread_cutoff,
            "Require a wider EMA7/EMA25 spread before entry.",
            {"parameter": "MIN_EMA_SPREAD_PCT", "direction": "increase_only", "suggested_min": round_value(spread_cutoff)},
        ),
        (
            "losses when candle body ratio too weak",
            losses["candle_body_ratio"] <= weak_body_cutoff,
            "Require stronger closed-candle body confirmation.",
            {"parameter": "MIN_CANDLE_BODY_RATIO", "direction": "increase_only", "suggested_min": weak_body_cutoff},
        ),
        (
            "losses when entry too far from EMA7",
            losses["distance_from_ema7_pct"] >= far_ema_cutoff,
            "Reject entries that are extended too far from EMA7.",
            {"parameter": "MAX_DISTANCE_FROM_EMA7_PCT", "direction": "decrease_only", "suggested_max": round_value(far_ema_cutoff)},
        ),
        (
            "losses after recent big candle / late entry",
            (losses["candle_body_ratio"] >= high_body_cutoff) & (losses["distance_from_ema7_pct"] >= 0.15),
            "Avoid chasing large candles when price is already extended from EMA7.",
            {"parameter": "AVOID_BIG_CANDLE_LATE_ENTRY", "direction": "enable_or_keep_strict", "suggested_only": True},
        ),
        (
            "losses during chop/range",
            (losses["ema_spread_abs_pct"] <= spread_cutoff) & (losses["candle_body_ratio"] <= 0.45),
            "Add a stricter no-trade chop filter when EMA spread is tight and bodies are soft.",
            {"parameter": "CHOP_FILTER", "direction": "enable_or_keep_strict", "suggested_only": True},
        ),
        (
            "losses after previous loss",
            losses["previous_trade_was_loss"],
            "Add or increase cooldown after a losing trade.",
            {"parameter": "COOLDOWN_AFTER_LOSS_CANDLES", "direction": "increase_only", "suggested_min": 1},
        ),
        (
            "losses during low volume periods",
            losses["volume"] <= low_volume_cutoff,
            "Require volume above the lower quartile of your trade sample.",
            {"parameter": "MIN_VOLUME_FILTER", "direction": "increase_only", "suggested_min": round_value(low_volume_cutoff)},
        ),
    ]

    patterns: list[dict[str, Any]] = []
    total_losses = len(losses)
    for name, mask, suggestion, recommendation in pattern_specs:
        matched = losses[mask]
        if matched.empty:
            continue
        patterns.append(
            {
                "pattern": name,
                "loss_count": int(len(matched)),
                "loss_share_pct": round_value((len(matched) / total_losses) * 100),
                "average_loss": round_value(matched["pnl_usdt"].mean()),
                "suggestion": suggestion,
                "recommendation": recommendation,
            }
        )
    patterns.sort(key=lambda item: (item["loss_count"], abs(item["average_loss"])), reverse=True)
    return patterns, thresholds


def build_recommendations(
    frame: pd.DataFrame,
    patterns: list[dict[str, Any]],
    min_trades: int,
) -> list[dict[str, Any]]:
    if len(frame) < min_trades:
        return []
    recommendations: list[dict[str, Any]] = []
    for pattern in patterns:
        recommendation = dict(pattern["recommendation"])
        recommendation["reason"] = pattern["pattern"]
        recommendation["suggested_only"] = True
        recommendation["safety"] = "Report only. Do not auto-apply."
        recommendations.append(recommendation)
    safe_recommendations = []
    for item in recommendations:
        parameter = str(item.get("parameter", "")).upper()
        if "LEVERAGE" in parameter or "ORDER_SIZE" in parameter:
            continue
        safe_recommendations.append(item)
    return safe_recommendations[:6]


def best_worst_groups(frame: pd.DataFrame, column: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = summarize_group(frame, column, limit=100)
    if not summary:
        return [], []
    best = sorted(summary, key=lambda row: (row["average_pnl"], row["win_rate_pct"], row["trades"]), reverse=True)[:3]
    worst = sorted(summary, key=lambda row: (row["average_pnl"], row["win_rate_pct"], -row["trades"]))[:3]
    return best, worst


def losing_examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    losses = frame[frame["pnl_usdt"] < 0].sort_values("pnl_usdt").head(5)
    examples: list[dict[str, Any]] = []
    for _, row in losses.iterrows():
        examples.append(
            {
                "trade_id": str(row.get("trade_id", "")),
                "timestamp": row["timestamp"].isoformat(),
                "side": str(row.get("side", "")),
                "pnl_usdt": round_value(row.get("pnl_usdt", 0)),
                "entry_reason": str(row.get("entry_reason", ""))[:140],
                "exit_reason": str(row.get("exit_reason", ""))[:140],
                "ema_spread_pct": round_value(row.get("ema_spread_pct", 0)),
                "body_ratio": round_value(row.get("candle_body_ratio", 0)),
                "distance_from_ema7_pct": round_value(row.get("distance_from_ema7_pct", 0)),
            }
        )
    return examples


def read_bot_log_notes() -> list[str]:
    notes: list[str] = []
    for path in BOT_LOG_PATHS:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
        except OSError:
            continue
        warnings = [line for line in lines if "WARNING" in line]
        errors = [line for line in lines if "ERROR" in line]
        if warnings or errors:
            notes.append(
                f"{path.name}: {len(warnings)} warnings and {len(errors)} errors in the latest {len(lines)} lines."
            )
        else:
            notes.append(f"{path.name}: no warnings/errors in the latest {len(lines)} lines.")
    if not notes:
        notes.append("No bot log file found yet. main.py now writes bot.log for future reviews.")
    return notes


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No data._"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    patterns: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    log_notes: list[str],
    min_trades: int,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    enough = len(frame) >= min_trades
    best_times, worst_times = best_worst_groups(frame, "time_window")
    best_setups, worst_setups = best_worst_groups(frame, "entry_reason")
    best_sides, worst_sides = best_worst_groups(frame, "side")

    top_winning_conditions = [
        best_times[0] if best_times else {},
        best_setups[0] if best_setups else {},
        best_sides[0] if best_sides else {},
    ]
    top_winning_conditions = [item for item in top_winning_conditions if item]

    report = [
        "# Learning Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "Safety: this report is suggestion-only. The bot does not auto-apply learning changes.",
        "",
        "## Summary",
        "",
        markdown_table([metrics]),
        "",
        f"Minimum trades before actionable learning: `{min_trades}`.",
        f"Actionable recommendation status: `{'enabled' if enough else 'waiting for more trades'}`.",
        "",
        "## Winning vs Losing Comparison",
        "",
        markdown_table(
            [
                {"feature": key, **value}
                for key, value in comparison.items()
            ]
        ),
        "",
        "## Top 3 Winning Conditions",
        "",
        markdown_table(top_winning_conditions[:3]),
        "",
        "## Top 3 Losing Conditions",
        "",
        markdown_table(
            [
                {
                    "pattern": item["pattern"],
                    "loss_count": item["loss_count"],
                    "loss_share_pct": item["loss_share_pct"],
                    "average_loss": item["average_loss"],
                    "suggestion": item["suggestion"],
                }
                for item in patterns[:3]
            ]
        ),
        "",
        "## Recommended Filter Adjustments",
        "",
    ]
    if recommendations:
        report.append(markdown_table(recommendations))
    else:
        report.append("_No actionable filter changes yet. Either there are not enough trades or no repeated loss pattern is clear._")
    report.extend(
        [
            "",
            "Important: recommendations may only make filters stricter. They must not increase leverage or order size.",
            "",
            "## Examples Of Losing Trades",
            "",
            markdown_table(examples),
            "",
            "## Suggested Rules To Avoid Similar Losses",
            "",
        ]
    )
    if patterns:
        for item in patterns[:5]:
            report.append(f"- {item['suggestion']} Reason: {item['pattern']} appeared in {item['loss_count']} losing trade(s).")
    else:
        report.append("- No losing-trade pattern detected yet.")
    report.extend(
        [
            "",
            "## Grouping Snapshot",
            "",
            "### By Date",
            markdown_table(summarize_group(frame, "date")),
            "",
            "### By Side",
            markdown_table(summarize_group(frame, "side")),
            "",
            "### By Entry Reason",
            markdown_table(summarize_group(frame, "entry_reason")),
            "",
            "### By Exit Reason",
            markdown_table(summarize_group(frame, "exit_reason")),
            "",
            "### By PnL Bucket",
            markdown_table(summarize_group(frame, "pnl_bucket")),
            "",
            "### By EMA Spread",
            markdown_table(summarize_group(frame, "ema_spread_bucket")),
            "",
            "### By Candle Body Ratio",
            markdown_table(summarize_group(frame, "body_ratio_bucket")),
            "",
            "### By Distance From EMA7",
            markdown_table(summarize_group(frame, "distance_ema7_bucket")),
            "",
            "### By Time Of Day",
            markdown_table(summarize_group(frame, "time_window")),
            "",
            "### By Holding Time",
            markdown_table(summarize_group(frame, "holding_time_bucket")),
            "",
            "## Best/Worst Time Windows",
            "",
            "Best:",
            markdown_table(best_times),
            "",
            "Worst:",
            markdown_table(worst_times),
            "",
            "## Best/Worst Setup Reasons",
            "",
            "Best:",
            markdown_table(best_setups),
            "",
            "Worst:",
            markdown_table(worst_setups),
            "",
            "## Bot Log Notes",
            "",
        ]
    )
    for note in log_notes:
        report.append(f"- {note}")
    report.append("")
    return "\n".join(report)


def build_state(
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    patterns: list[dict[str, Any]],
    thresholds: dict[str, float],
    recommendations: list[dict[str, Any]],
    min_trades: int,
    auto_apply_learning: bool,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suggested_only": True,
        "auto_apply_learning_requested": auto_apply_learning,
        "auto_apply_learning_disabled_for_safety": True,
        "minimum_trades_before_learning": min_trades,
        "enough_trades_for_learning": len(frame) >= min_trades,
        "metrics": metrics,
        "detected_losing_patterns": patterns,
        "pattern_thresholds_used": thresholds,
        "recommended_parameter_changes": recommendations,
        "safety_constraints": [
            "Do not auto-apply recommendations.",
            "Do not increase leverage from learning.",
            "Do not increase order size from learning.",
            "Only stricter filters may be recommended.",
        ],
    }


def empty_outputs(min_trades: int, auto_apply_learning: bool) -> tuple[str, dict[str, Any]]:
    metrics = {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0,
        "average_win": 0,
        "average_loss": 0,
        "gross_profit": 0,
        "gross_loss": 0,
        "profit_factor": 0,
        "max_consecutive_losses": 0,
    }
    state = build_state(
        frame=pd.DataFrame(),
        metrics=metrics,
        patterns=[],
        thresholds={},
        recommendations=[],
        min_trades=min_trades,
        auto_apply_learning=auto_apply_learning,
    )
    report = "\n".join(
        [
            "# Learning Report",
            "",
            f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "No trades were found in `trades.csv` yet.",
            "",
            "Safety: this module is suggestion-only and does not change live trading rules.",
            "",
            f"Minimum trades before actionable learning: `{min_trades}`.",
        ]
    )
    return report, state


def write_suggested_upgrade_files(state: dict[str, Any]) -> None:
    recommendations = state.get("recommended_parameter_changes", [])
    patterns = state.get("detected_losing_patterns", [])
    generated_at = state.get("generated_at", datetime.now(timezone.utc).isoformat())

    env_lines = [
        "# Suggested .env update for MoneyMaker",
        f"# Generated at: {generated_at}",
        "# Safety: review manually before applying. This file is never auto-applied.",
        "# Learning may only recommend stricter filters. It must not increase leverage or order size.",
        "",
    ]
    if recommendations:
        for item in recommendations:
            env_lines.extend(format_env_recommendation(item))
    else:
        env_lines.append("# No .env filter update is recommended yet.")
    SUGGESTED_ENV_UPDATE_PATH.write_text("\n".join(env_lines).rstrip() + "\n", encoding="utf-8")

    strategy_lines = [
        "# Suggested Strategy Update",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "Safety: these are patch notes only. No `.env` or `main.py` changes were applied.",
        "",
        "## Top Losing Patterns",
        "",
    ]
    if patterns:
        for pattern in patterns[:5]:
            strategy_lines.append(
                f"- {pattern.get('pattern', 'pattern')}: {pattern.get('loss_count', 0)} loss(es), "
                f"{pattern.get('loss_share_pct', 0)}% of losses. {pattern.get('suggestion', 'Review manually.')}"
            )
    else:
        strategy_lines.append("- No repeated losing pattern is clear yet.")

    strategy_lines.extend(["", "## Suggested Stricter Filters", ""])
    if recommendations:
        for item in recommendations:
            parameter = item.get("parameter", "filter")
            direction = item.get("direction", "review")
            reason = item.get("reason", "Review this filter manually.")
            value = item.get("suggested_min", item.get("suggested_max", "manual review"))
            strategy_lines.append(f"- `{parameter}` `{direction}` -> `{value}` because {reason}.")
    else:
        strategy_lines.append("- No filter change suggested yet.")

    strategy_lines.extend(
        [
            "",
            "## Approval Requirement",
            "",
            "- Human approval is required before applying any suggested `.env` setting.",
            "- Do not increase leverage or order size from learning output.",
            "- Do not loosen filters automatically.",
        ]
    )
    SUGGESTED_STRATEGY_UPDATE_PATH.write_text("\n".join(strategy_lines).rstrip() + "\n", encoding="utf-8")


def format_env_recommendation(item: dict[str, Any]) -> list[str]:
    parameter = str(item.get("parameter", "")).strip().upper()
    direction = item.get("direction", "review")
    reason = item.get("reason", "Review manually.")
    if not parameter or "LEVERAGE" in parameter or "ORDER_SIZE" in parameter:
        return []

    lines = [
        f"# {reason}",
        f"# Direction: {direction}",
    ]
    if "suggested_min" in item:
        lines.append(f"{parameter}={item['suggested_min']}")
    elif "suggested_max" in item:
        lines.append(f"{parameter}={item['suggested_max']}")
    else:
        lines.append(f"# {parameter}=manual_review_required")
    lines.append("")
    return lines


def total_pnl_from_metrics(metrics: dict[str, Any]) -> float:
    gross_profit = safe_float(metrics.get("gross_profit", 0))
    gross_loss = safe_float(metrics.get("gross_loss", 0))
    return round_value(gross_profit - gross_loss)


def summarize_patterns_for_discord(state: dict[str, Any]) -> str:
    patterns = state.get("detected_losing_patterns", [])
    if not patterns:
        return "none detected"
    return "; ".join(
        f"{item.get('pattern', 'pattern')} ({item.get('loss_count', 0)} losses)"
        for item in patterns[:3]
    )


def summarize_recommendations_for_discord(state: dict[str, Any]) -> str:
    recommendations = state.get("recommended_parameter_changes", [])
    if not recommendations:
        return "none"
    parts = []
    for item in recommendations[:3]:
        parameter = item.get("parameter", "filter")
        value = item.get("suggested_min", item.get("suggested_max", "manual review"))
        parts.append(f"{parameter} -> {value}")
    return "; ".join(parts)


def safe_float(value: Any) -> float:
    try:
        if value == "inf":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def send_discord_embed(
    title: str,
    fields: list[dict[str, Any]] | None = None,
    description: str = "",
    color: int = DISCORD_LEARNING_COLOR,
) -> None:
    webhook_url = normalize_optional_value(os.getenv("DISCORD_WEBHOOK_URL"))
    if not webhook_url:
        return

    embed: dict[str, Any] = {
        "title": title,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {
            "text": "MoneyMaker learning module | suggestion-only, never auto-applied",
        },
    }
    if description:
        embed["description"] = description[:3900]
    if fields:
        embed["fields"] = fields[:25]

    try:
        response = requests.post(
            webhook_url,
            json={
                "username": "MoneyMaker",
                "allowed_mentions": {"parse": []},
                "embeds": [embed],
            },
            timeout=DISCORD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.getLogger("trade_analyzer").error("Discord learning alert failed: %s", exc)


def send_analysis_started_alert(min_trades: int, auto_apply_learning: bool) -> None:
    send_discord_embed(
        title="🧠 Learning analysis started",
        fields=[
            discord_field("Minimum Trades", min_trades, inline=True),
            discord_field("Auto Apply", "disabled for safety" if auto_apply_learning else "false", inline=True),
            discord_field("Report", "learning_report.md", inline=True),
        ],
        description="Reading trades.csv and bot logs for suggestion-only review.",
        color=DISCORD_LEARNING_COLOR,
    )


def send_analysis_completed_alert(state: dict[str, Any]) -> None:
    metrics = state.get("metrics", {})
    recommendations = state.get("recommended_parameter_changes", [])
    insights = top_learning_insights(state)
    total_pnl = total_pnl_from_metrics(metrics)
    description = "No top insights detected yet."
    if insights:
        description = "\n".join(f"{index}. {insight}" for index, insight in enumerate(insights, start=1))

    send_discord_embed(
        title="🧠 Learning analysis completed",
        fields=[
            discord_field("Total Trades", metrics.get("total_trades", 0), inline=True),
            discord_field("Win Rate", f"{metrics.get('win_rate_pct', 0)}%", inline=True),
            discord_field("Profit Factor", metrics.get("profit_factor", 0), inline=True),
            discord_field("Total PnL", total_pnl, inline=True),
            discord_field("Suggestions Exist", "yes" if recommendations else "no", inline=True),
            discord_field("Env Update Recommended", "yes" if recommendations else "no", inline=True),
            discord_field("Top Losing Patterns", summarize_patterns_for_discord(state), inline=False),
            discord_field("Suggested Stricter Filters", summarize_recommendations_for_discord(state), inline=False),
        ],
        description=description,
        color=DISCORD_LEARNING_COLOR if recommendations else DISCORD_WARNING_COLOR,
    )
    if recommendations:
        send_discord_embed(
            title="Env update suggested. Review suggested_env_update.txt",
            fields=[
                discord_field("Suggested Env File", "suggested_env_update.txt", inline=True),
                discord_field("Strategy Notes", "suggested_strategy_update.md", inline=True),
                discord_field("Auto Applied", "false", inline=True),
            ],
            description="Human approval is required before applying any learning suggestion.",
            color=DISCORD_WARNING_COLOR,
        )


def top_learning_insights(state: dict[str, Any]) -> list[str]:
    patterns = state.get("detected_losing_patterns", [])
    insights: list[str] = []
    for pattern in patterns[:3]:
        name = pattern.get("pattern", "Losing pattern")
        loss_count = pattern.get("loss_count", 0)
        loss_share = pattern.get("loss_share_pct", 0)
        suggestion = pattern.get("suggestion", "Review this setup.")
        insights.append(f"{name}: {loss_count} loss(es), {loss_share}% of losses. {suggestion}")

    if insights:
        return insights[:3]

    recommendations = state.get("recommended_parameter_changes", [])
    for recommendation in recommendations[:3]:
        parameter = recommendation.get("parameter", "filter")
        reason = recommendation.get("reason", "Review suggested filter.")
        insights.append(f"{parameter}: {reason}")
    return insights[:3]


def discord_field(name: str, value: Any, inline: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "value": f"`{str(value)[:DISCORD_MAX_FIELD_VALUE_LENGTH - 2]}`",
        "inline": inline,
    }


def normalize_optional_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def main() -> int:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    min_trades = int(os.getenv("MIN_TRADES_BEFORE_LEARNING", "20"))
    auto_apply_learning = parse_bool(os.getenv("AUTO_APPLY_LEARNING"), False)
    analysis_discord_alert = parse_bool(os.getenv("ANALYSIS_DISCORD_ALERT"), True)
    if auto_apply_learning:
        print("Auto-apply learning is disabled for safety.")

    if analysis_discord_alert:
        send_analysis_started_alert(min_trades=min_trades, auto_apply_learning=auto_apply_learning)

    frame = load_trades()
    if frame.empty:
        report, state = empty_outputs(min_trades, auto_apply_learning)
    else:
        metrics = calculate_metrics(frame)
        comparison = compare_winners_losers(frame)
        patterns, thresholds = identify_losing_patterns(frame)
        recommendations = build_recommendations(frame, patterns, min_trades)
        examples = losing_examples(frame)
        log_notes = read_bot_log_notes()
        report = build_report(
            frame=frame,
            metrics=metrics,
            comparison=comparison,
            patterns=patterns,
            recommendations=recommendations,
            examples=examples,
            log_notes=log_notes,
            min_trades=min_trades,
        )
        state = build_state(
            frame=frame,
            metrics=metrics,
            patterns=patterns,
            thresholds=thresholds,
            recommendations=recommendations,
            min_trades=min_trades,
            auto_apply_learning=auto_apply_learning,
        )

    REPORT_PATH.write_text(report, encoding="utf-8")
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    write_suggested_upgrade_files(state)
    if analysis_discord_alert:
        send_analysis_completed_alert(state)
    print(f"Learning report written to {REPORT_PATH}")
    print(f"Learning state written to {STATE_PATH}")
    print(f"Suggested env update written to {SUGGESTED_ENV_UPDATE_PATH}")
    print(f"Suggested strategy update written to {SUGGESTED_STRATEGY_UPDATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
