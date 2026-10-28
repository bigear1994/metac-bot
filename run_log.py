"""Run journal for the myara metac bot.

Every invocation of main.py appends one JSONL file under ../runs/ containing:
- run_start  : mode, publish flag, targets
- question   : one line per question (url, type, outcome, price_estimate, minutes)
- usage      : aggregated litellm token/cost numbers captured via success_callback
- run_end    : totals + exit-code-relevant summary

Designed to never crash the bot: every extraction is defensive.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# ---- litellm usage capture -------------------------------------------------
# litellm.success_callback entries are called as fn(kwargs, response,
# start_time, end_time) for every completed call. We aggregate totals here and
# attach them at run end. Registration happens in start_run().
_usage_events: list[dict[str, Any]] = []


def _litellm_usage_callback(kwargs, completion_response, start_time, end_time):
    try:
        usage = getattr(completion_response, "usage", None)
        model = kwargs.get("litellm_params", {}).get("model") or kwargs.get(
            "model", "unknown"
        )
        tokens = getattr(usage, "total_tokens", None)
        cost = None
        hidden = getattr(completion_response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
        _usage_events.append(
            {
                "model": str(model),
                "total_tokens": int(tokens) if tokens is not None else None,
                "cost_usd": cost,
            }
        )
    except Exception:  # never break a live call on accounting
        pass


def register_usage_callback() -> None:
    try:
        import litellm

        cb = _litellm_usage_callback
        if cb not in litellm.success_callback:
            litellm.success_callback.append(cb)
    except Exception:
        pass


# ---- JSONL writing ----------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def start_run(mode: str, publish: bool, targets: list) -> dict:
    register_usage_callback()
    path = RUNS_DIR / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}.jsonl"
    _append(
        path,
        {
            "event": "run_start",
            "ts": _utcnow(),
            "mode": mode,
            "publish": publish,
            "targets": [str(t) for t in targets],
        },
    )
    return {"path": path, "t0": time.monotonic()}


def _prediction_summary(prediction: Any) -> str:
    try:
        text = str(prediction)
        return text[:280]
    except Exception:
        return "<unprintable>"


def record_reports(run: dict, reports: list) -> None:
    from forecasting_tools import ForecastReport

    path = run["path"]
    for r in reports:
        if isinstance(r, BaseException):
            _append(
                path,
                {
                    "event": "question",
                    "ts": _utcnow(),
                    "ok": False,
                    "error": f"{type(r).__name__}: {r}"[:500],
                },
            )
            continue
        if not isinstance(r, ForecastReport):
            _append(
                path,
                {
                    "event": "question",
                    "ts": _utcnow(),
                    "ok": None,
                    "error": f"unexpected report type {type(r).__name__}",
                },
            )
            continue
        q = r.question
        _append(
            path,
            {
                "event": "question",
                "ts": _utcnow(),
                "ok": True,
                "url": getattr(q, "page_url", None),
                "question_type": type(q).__name__,
                "prediction": _prediction_summary(r.prediction),
                "price_estimate_usd": r.price_estimate,
                "minutes_taken": r.minutes_taken,
                "errors": len(r.errors or []),
                "explanation_chars": len(r.explanation or ""),
            },
        )


def attach_model_usage(run: dict) -> None:
    """Write aggregated litellm usage (tokens/cost per model) if available."""
    if not _usage_events:
        return
    totals: dict[str, dict[str, float]] = {}
    for e in _usage_events:
        bucket = totals.setdefault(e["model"], {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        bucket["calls"] += 1
        if e["total_tokens"]:
            bucket["tokens"] += e["total_tokens"]
        if e["cost_usd"]:
            bucket["cost_usd"] += e["cost_usd"]
    _append(
        run["path"],
        {
            "event": "usage",
            "ts": _utcnow(),
            "models": totals,
            "total_cost_usd": round(sum(v["cost_usd"] for v in totals.values()), 6),
        },
    )


def finish_run(run: dict, reports: list) -> int:
    """Write run_end and return the process exit code (0 ok, 1 hard-fail)."""
    from forecasting_tools import ForecastReport

    valid = [r for r in reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in reports if isinstance(r, BaseException)]
    cost = sum(r.price_estimate or 0 for r in valid)
    rec = {
        "event": "run_end",
        "ts": _utcnow(),
        "seconds": round(time.monotonic() - run["t0"], 1),
        "forecast_ok": len(valid),
        "forecast_failed": len(exceptions),
        "sum_price_estimate_usd": round(cost, 6),
    }
    _append(run["path"], rec)
    print(f"[run_log] {run['path'].name}: {rec}")
    # Hard failure only when we attempted questions and everything blew up.
    if reports and not valid:
        return 1
    return 0
