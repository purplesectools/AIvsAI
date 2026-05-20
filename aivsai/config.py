"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("AIVSAI_HOST", "127.0.0.1")
    port: int = _get_int("AIVSAI_PORT", 8000)

    default_attacker_base_url: str = os.getenv(
        "AIVSAI_DEFAULT_ATTACKER_BASE_URL", "http://localhost:11434/v1"
    )
    default_attacker_model: str = os.getenv(
        "AIVSAI_DEFAULT_ATTACKER_MODEL", "llama3.1:8b"
    )
    default_attacker_api_key: str = os.getenv("AIVSAI_DEFAULT_ATTACKER_API_KEY", "")

    max_iterations: int = _get_int("AIVSAI_MAX_ITERATIONS", 50)
    max_requests_per_second: float = _get_float("AIVSAI_MAX_REQUESTS_PER_SECOND", 2.0)
    # Per-HTTP-call timeout. Heavy local models (mixtral 8x7b, llama 70b) on
    # CPU/MPS Macs can take 60-120s per call, especially the first call after
    # the model is loaded into RAM. 180s is a safer default than 60.
    request_timeout_seconds: float = _get_float("AIVSAI_REQUEST_TIMEOUT_SECONDS", 180.0)

    # SSRF guard. Block private / loopback / link-local / cloud-metadata
    # IP destinations by default. Pen-testers explicitly testing internal
    # systems can either set this env var globally or tick a per-run
    # checkbox in the UI ("I'm testing an internal/local target").
    allow_private_targets: bool = os.getenv("AIVSAI_ALLOW_PRIVATE_TARGETS", "0") not in ("0", "false", "False", "")


settings = Settings()
