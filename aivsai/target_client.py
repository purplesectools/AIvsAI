"""Sends prompts to the user-supplied target LLM endpoint.

The user describes their endpoint as an HTTP request template: URL, method,
headers, and a body where the literal token ``{{PROMPT}}`` marks where the
attacker's payload should go. We deep-substitute the token and POST it.

Response extraction supports a simple dotted/indexed path syntax:
    choices[0].message.content
    data.reply
    output.text
If no path is given (or the path doesn't resolve) we fall back to returning
the entire body as a JSON string so the attacker still has *something* to
react to.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Optional

import httpx

from .config import settings
from .models import TargetConfig
from .ssrf import check_target_url


PROMPT_TOKEN = "{{PROMPT}}"


def _strip_header(headers: dict, name: str) -> None:
    """Remove a header case-insensitively (lets httpx auto-set Content-Type)."""
    for k in [h for h in headers if h.lower() == name.lower()]:
        del headers[k]


def _substitute(node: Any, prompt: str) -> Any:
    """Deep-replace ``{{PROMPT}}`` inside the request body."""
    if isinstance(node, str):
        return node.replace(PROMPT_TOKEN, prompt)
    if isinstance(node, list):
        return [_substitute(x, prompt) for x in node]
    if isinstance(node, dict):
        return {k: _substitute(v, prompt) for k, v in node.items()}
    return node


_INDEX_RE = re.compile(r"\[(-?\d+)\]")


def _extract(body: Any, path: Optional[str]) -> str:
    """Extract a string from the response body using a simple dotted path."""
    if not path:
        return _to_text(body)

    cur: Any = body
    # Split on '.' but preserve `foo[0]` style indices.
    for raw in path.split("."):
        if cur is None:
            break
        # Pull off a key, then any number of [i] indices.
        match = re.match(r"([^\[]+)?(.*)", raw)
        if not match:
            continue
        key, idx_part = match.groups()
        if key:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return _to_text(body)  # path didn't fit; bail to raw body
        for m in _INDEX_RE.finditer(idx_part or ""):
            i = int(m.group(1))
            if isinstance(cur, list) and -len(cur) <= i < len(cur):
                cur = cur[i]
            else:
                return _to_text(body)
    return _to_text(cur)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


class TargetClient:
    """Stateless wrapper around the user's target endpoint."""

    def __init__(self, config: TargetConfig):
        self.config = config
        # HTTP/2 if `h2` is installed; fall back to HTTP/1.1 cleanly otherwise.
        try:
            self._client = httpx.AsyncClient(
                timeout=settings.request_timeout_seconds,
                trust_env=False, http2=True,
            )
        except ImportError:
            self._client = httpx.AsyncClient(
                timeout=settings.request_timeout_seconds,
                trust_env=False, http2=False,
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "TargetClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def send(self, prompt: str) -> dict:
        """Send a prompt and return a structured result for the UI/attacker.

        Always returns a dict so the orchestrator can stream it; never raises
        on HTTP errors — callers want to see the failure as part of the run.
        """
        # SSRF guard — block private/loopback/cloud-metadata targets unless
        # the user has explicitly opted in (per-run checkbox or env var).
        ssrf_err = check_target_url(
            self.config.url,
            allow_private=bool(self.config.allow_private) or settings.allow_private_targets,
        )
        if ssrf_err:
            return {"ok": False, "status": 0, "raw": "",
                    "extracted": f"[ssrf blocked] {ssrf_err}"}

        body = _substitute(copy.deepcopy(self.config.body), prompt)
        # Allow header values to use ``{{PROMPT}}`` too (rare but supported).
        headers = {k: _substitute(v, prompt) for k, v in (self.config.headers or {}).items()}

        # Defensive defaults — some edges (CDNs, HF routers) reject requests
        # without a User-Agent or with the default python-httpx UA. We only
        # add these when the user hasn't already set them.
        if not any(k.lower() == "user-agent" for k in headers):
            headers["User-Agent"] = "AIvsAI/0.1 (+https://github.com/aivsai)"
        if not any(k.lower() == "accept" for k in headers):
            headers["Accept"] = "application/json, text/plain;q=0.9, */*;q=0.5"

        method = self.config.method.upper()
        request_kwargs: dict = {"headers": headers}

        if method in {"POST", "PUT", "PATCH"}:
            bt = (self.config.body_type or "json").lower()
            if bt == "json":
                request_kwargs["json"] = body
            elif bt == "form":
                # urlencoded — body should be a flat dict
                if not isinstance(body, dict):
                    return {"ok": False, "status": 0, "raw": "",
                            "extracted": "[config error] body_type=form requires a flat dict body."}
                request_kwargs["data"] = {k: str(v) for k, v in body.items()}
                # httpx sets Content-Type for us
                _strip_header(headers, "content-type")
            elif bt == "multipart":
                if not isinstance(body, dict):
                    return {"ok": False, "status": 0, "raw": "",
                            "extracted": "[config error] body_type=multipart requires a flat dict body."}
                # httpx wants {field: (None, value)} for non-file fields, so the
                # boundary is generated correctly and Content-Type set automatically.
                request_kwargs["files"] = {k: (None, str(v)) for k, v in body.items()}
                _strip_header(headers, "content-type")
            elif bt == "raw":
                if isinstance(body, (dict, list)):
                    return {"ok": False, "status": 0, "raw": "",
                            "extracted": "[config error] body_type=raw requires a string body."}
                request_kwargs["content"] = body if isinstance(body, (str, bytes)) else str(body)
            else:
                return {"ok": False, "status": 0, "raw": "",
                        "extracted": f"[config error] unknown body_type {bt!r}."}

        try:
            resp = await self._client.request(method, self.config.url, **request_kwargs)
        except httpx.HTTPError as e:
            return {
                "ok": False,
                "status": 0,
                "raw": "",
                "extracted": f"[network error] {e!r}",
            }

        raw_text = resp.text
        try:
            parsed = resp.json()
        except (ValueError, json.JSONDecodeError):
            parsed = raw_text

        extracted = _extract(parsed, self.config.response_path)
        return {
            "ok": resp.is_success,
            "status": resp.status_code,
            "raw": raw_text[:8000],  # cap to keep UI snappy
            "extracted": extracted,
        }
