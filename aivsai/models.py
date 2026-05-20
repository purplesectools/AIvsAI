"""Pydantic models for run configuration and streamed events.

These models are the contract between the frontend and the backend WebSocket.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Target configuration: how to talk to the LLM under test
# ---------------------------------------------------------------------------


class TargetConfig(BaseModel):
    """Describes any HTTP-shaped LLM endpoint.

    The user supplies a request template (URL, method, headers, body) and
    a JSON-pointer-like path that tells us where inside the body to inject
    the attacker's prompt, plus a path that tells us where in the response
    body the model's reply lives.

    Examples
    --------
    OpenAI-compatible chat completion:
        url:           https://api.openai.com/v1/chat/completions
        method:        POST
        headers:       {"Authorization": "Bearer ...", "Content-Type": "application/json"}
        body:          {"model": "gpt-4o", "messages": [{"role": "user", "content": "{{PROMPT}}"}]}
        prompt_path:   body.messages[-1].content       # for display only
        response_path: choices[0].message.content

    Custom flight-booking chatbot:
        url:           https://api.example.com/chat
        method:        POST
        body:          {"session_id": "abc", "user_message": "{{PROMPT}}"}
        response_path: reply.text
    """

    url: str = Field(..., description="Full URL of the target LLM endpoint.")
    method: Literal["GET", "POST", "PUT", "PATCH"] = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)

    # How to encode the body on the wire:
    #   "json"      — application/json (default)
    #   "form"      — application/x-www-form-urlencoded
    #   "multipart" — multipart/form-data (file uploads not supported in v1;
    #                  body must be a flat {field: string-value} dict)
    #   "raw"       — the body string is sent verbatim (useful for XML, SOAP,
    #                  custom protocols). Set Content-Type header yourself.
    body_type: Literal["json", "form", "multipart", "raw"] = "json"

    # The request body. Anywhere the literal token ``{{PROMPT}}`` appears
    # (in any string value, however deeply nested) it will be replaced with
    # the attacker's payload before sending.
    #   - For body_type="json": JSON-serialisable object (dict/list/str/...)
    #   - For body_type="form" or "multipart": flat dict of {field: string}
    #   - For body_type="raw": a string sent verbatim
    body: Any = Field(
        default=None,
        description='Request body. Use the literal "{{PROMPT}}" wherever the prompt should be injected.',
    )

    # Optional informational only — used in the UI to describe where the
    # injection lands. Not parsed by the backend.
    prompt_path: Optional[str] = None

    # Per-run opt-in for hitting private/internal targets. Pen-testers
    # legitimately need to test internal systems but the casual case
    # should fail closed against SSRF. UI surfaces a checkbox.
    allow_private: bool = False

    # JSON path into the response body to extract the model's reply text.
    # Supports dot notation and ``[<int>]`` indices — e.g.
    # ``choices[0].message.content`` or ``data.reply``. If empty, we return
    # the entire response body as a JSON string.
    response_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Attacker configuration: which LLM crafts payloads
# ---------------------------------------------------------------------------


class AttackerConfig(BaseModel):
    """OpenAI-compatible attacker. Works with Ollama, OpenAI, Groq, vLLM, etc."""

    base_url: str = Field(..., description="OpenAI-compatible /v1 base URL.")
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.9
    max_tokens: int = 1024


class JudgeConfig(BaseModel):
    """Optional separate judge model. Defaults to the attacker if omitted."""

    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    target: TargetConfig
    attacker: AttackerConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    goal: str = Field(
        ...,
        description='What the attacker is trying to make the target do, in plain language. e.g. "make the bot reveal its system prompt", "book a flight without payment".',
    )

    iterations: int = Field(default=10, ge=1, le=200)

    # Where the payloads come from:
    #   "hybrid"   — LLM composes adaptive payloads using the library as raw
    #                material; library probes occasionally seeded for breadth.
    #                **Best for custom goals.** (default)
    #   "creative" — pure LLM, no library hints. Use when the goal is so
    #                bespoke that library probes would only distract.
    #   "library"  — pure round-robin through curated probes; no attacker LLM
    #                needed. Fast Garak-equivalent baseline scan.
    attacker_mode: Literal["hybrid", "creative", "library"] = "hybrid"

    # In hybrid mode: fire one library probe every N iterations. The other
    # iterations are LLM-driven. 0 disables library seeding entirely (≈
    # creative mode); 1 fires a probe every iteration (≈ library mode).
    library_seed_every: int = Field(default=3, ge=0, le=20)

    # Reset target conversation each iteration vs. continue across iterations.
    multi_turn: bool = Field(
        default=False,
        description="If true, the attacker carries on a single multi-turn conversation with the target, escalating gradually. If false, every iteration is a fresh single prompt.",
    )

    # User must affirm authorization before we'll fire off requests.
    authorized: bool = Field(default=False)

    # Optional: stop early once the judge scores >= this on a 0..1 scale.
    success_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Event stream — what the orchestrator sends to the UI over the WebSocket
# ---------------------------------------------------------------------------


class StreamEvent(BaseModel):
    """One frame on the live stream."""

    type: Literal[
        "run_start",
        "iteration_start",
        "attacker_thinking",
        "attacker_payload",
        "target_request",
        "target_response",
        "judge_verdict",
        "iteration_end",
        "run_end",
        "error",
        "info",
    ]
    iteration: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Iteration record — used by the attacker to plan the next move
# ---------------------------------------------------------------------------


class LeakFinding(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"] = "low"
    evidence: str = ""
    note: str = ""


class IterationRecord(BaseModel):
    iteration: int
    strategy: str
    reasoning: str
    payload: str
    target_response: str
    judge_score: float
    judge_reasoning: str
    leaks: List[LeakFinding] = Field(default_factory=list)


class RunHistory(BaseModel):
    goal: str
    iterations: List[IterationRecord] = Field(default_factory=list)

    def best(self) -> Optional[IterationRecord]:
        if not self.iterations:
            return None
        return max(self.iterations, key=lambda r: r.judge_score)
