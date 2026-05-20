"""Per-run persistence — every iteration's events streamed to disk.

Layout:
  runs/
    2026-05-08T14-12-33Z__reveal-system-prompt__abc123/
      events.jsonl       # every StreamEvent in order
      summary.json       # final summary written on run_end
      config.json        # the RunConfig (with API keys masked)

Each event is a single JSON object on its own line — `tail -f events.jsonl`
works while a run is in progress. The run id (folder name) is the slugified
goal + a short random suffix so listings sort newest-first by timestamp.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import secrets
from pathlib import Path
from typing import Any, Optional


# Where runs go. Override with AIVSAI_RUNS_DIR.
import os
RUNS_DIR = Path(os.getenv("AIVSAI_RUNS_DIR", "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return (s[:max_len] or "run")


def _mask_secrets(cfg: dict) -> dict:
    """Don't write API keys to disk. Mutates a copy."""
    out = json.loads(json.dumps(cfg))  # cheap deep copy
    for path in (
        ("attacker", "api_key"),
        ("judge", "api_key"),
    ):
        cur = out
        for key in path[:-1]:
            cur = cur.get(key, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict) and path[-1] in cur:
            cur[path[-1]] = "***redacted***"
    # Mask Authorization / Cookie / x-api-key headers in target.headers
    headers = (out.get("target") or {}).get("headers") or {}
    for k in list(headers.keys()):
        if k.lower() in ("authorization", "cookie", "x-api-key", "x-auth-token", "api-key"):
            headers[k] = "***redacted***"
    return out


class RunRecorder:
    """Append-only JSONL writer for a single run's events.

    Use as an async context manager. The writer always closes its file
    even on crash, and writes summary.json once on shutdown.
    """

    def __init__(self, goal: str, config_dict: dict):
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.run_id = f"{ts}__{_slug(goal)}__{secrets.token_hex(3)}"
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        # Write the (masked) config as the first artifact
        with (self.dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(_mask_secrets(config_dict), f, indent=2)

        self._events_path = self.dir / "events.jsonl"
        self._fp = None
        self._lock = asyncio.Lock()
        self._summary: dict[str, Any] = {
            "run_id": self.run_id,
            "goal": goal,
            "started_at": ts,
            "iterations": 0,
            "best_score": 0.0,
            "leaks_total": 0,
            "status": "running",
        }

    async def __aenter__(self) -> "RunRecorder":
        # Open lazily so we don't claim a file handle until first write
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def write(self, event: dict) -> None:
        async with self._lock:
            if self._fp is None:
                self._fp = self._events_path.open("a", encoding="utf-8", buffering=1)
            self._fp.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fp.flush()

            # Update rolling summary
            t = event.get("type")
            data = event.get("data") or {}
            if t == "iteration_start":
                self._summary["iterations"] = max(self._summary["iterations"], event.get("iteration", 0))
            elif t == "judge_verdict":
                score = float(data.get("score", 0.0))
                if score > self._summary["best_score"]:
                    self._summary["best_score"] = score
                self._summary["leaks_total"] += len(data.get("leaks") or [])
            elif t == "run_end":
                self._summary["status"] = "completed"
                self._summary["best_score"] = float(data.get("best_score", self._summary["best_score"]))
                self._summary["achieved"] = bool(data.get("achieved", False))
                self._summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    async def close(self) -> None:
        async with self._lock:
            if self._fp:
                self._fp.close()
                self._fp = None
            self._summary.setdefault("finished_at",
                datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"))
            if self._summary["status"] == "running":
                self._summary["status"] = "interrupted"
            with (self.dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(self._summary, f, indent=2)


# ---------------------------------------------------------------- API helpers


def list_runs(limit: int = 50) -> list[dict]:
    """Return summaries of recent runs, newest first."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            with sp.open("r", encoding="utf-8") as f:
                s = json.load(f)
            s["run_id"] = d.name
            runs.append(s)
        except (OSError, ValueError):
            continue
        if len(runs) >= limit:
            break
    return runs


def get_run(run_id: str) -> Optional[dict]:
    """Return full details for a run: summary, masked config, all events."""
    if not re.match(r"^[\w.\-:T]+$", run_id):
        return None
    d = RUNS_DIR / run_id
    if not d.is_dir():
        return None
    out: dict[str, Any] = {"run_id": run_id}
    for name in ("summary.json", "config.json"):
        p = d / name
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    out[name.split(".")[0]] = json.load(f)
            except (OSError, ValueError):
                pass
    events: list[dict] = []
    ep = d / "events.jsonl"
    if ep.exists():
        try:
            with ep.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
    out["events"] = events
    return out
