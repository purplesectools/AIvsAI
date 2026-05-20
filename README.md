# AI vs AI

**Adaptive adversarial testing for LLM applications.**

AI vs AI is an open-source red-team tool that points one LLM (the *attacker*)
at another LLM (the *target*) and runs a closed-loop attack: each iteration
the attacker reads the target's last response, picks a strategy, mutates its
payload, and tries again — driving toward a goal you define. A separate
*judge* model scores whether the goal was reached.

Unlike static-payload tools, every payload is generated **at runtime** by the
attacker LLM, so the attacks adapt to whatever guardrails or system prompt
your target ships.

AI vs AI focuses on adaptive adversarial orchestration rather than static
jailbreak payloads. The attacker changes strategies dynamically based on
prior failures, judge scores, and target behavior.

---

## Dashboard

![AI vs AI Dashboard](docs/images/hero-dashboard.png)

---

## Adaptive Attack Evolution

AI vs AI dynamically changes attack strategies based on target behavior and
scoring feedback.

Examples include:
- authority impersonation
- instruction override
- fake tool outputs
- role-play jailbreaks
- context injection
- forced pivots after plateau detection

![Adaptive Reasoning](docs/images/adaptive-reasoning.png)

---

## Real-world Target Support

AI vs AI can test arbitrary HTTP/cURL-based LLM applications by importing
raw requests directly

Supports:
- cookies
- auth headers
- JSON APIs
- custom headers
- OpenAI-compatible endpoints


![Quick Paste Parser](docs/images/quickpaste-parser.png)


---

## Why another red-team tool?

Tools like Garak and PyRIT are excellent foundations for LLM security testing. AI vs AI focuses specifically on adaptive closed-loop adversarial orchestration, where attacks evolve dynamically based on prior target responses and judge feedback.

AI vs AI is built around three ideas:

1. **Closed-loop, not catalog.** The attacker adapts its payload strategy based on what actually worked or failed in prior rounds.based on what actually worked or failed in prior rounds.

2. **Bring-any-target.** The target is described as a generic HTTP request template with a `{{PROMPT}}` token, so it works against OpenAI-compatible APIs, internal LLM gateways, RAG apps, custom chatbots, or anything in between.

3. **Bring-any-attacker.** The attacker can be any OpenAI-compatible endpoint — local Ollama by default, or hosted APIs (OpenAI, Groq, vLLM, OpenRouter, etc.) when more capable reasoning models are needed.
---

## Features

- Adaptive multi-iteration attack orchestration
- Hybrid attack mode (LLM reasoning + curated payload probes)
- Real-time attack reasoning stream
- Dynamic tactic pivoting based on score feedback
- OpenAI-compatible inference API support
- Ollama/local attacker model support
- Arbitrary HTTP/cURL request testing
- Burp-style request parsing
- Live verdict scoring and attack history
- JSON export for runs

---

## Quickstart

### Local (Python)

```bash
# 1. Install
git clone <this repo>
cd AIvsAI
python -m venv .venv && source .venv/bin/activate
pip install .

# 2. Pull a local attacker model with Ollama
ollama pull llama3.1:8b

# 3. Launch
aivsai
# → http://127.0.0.1:8000
```



---

## How to use it

Open the UI and fill in three things:

### 1. Target endpoint

Describe how to call the LLM you want to test. AI vs AI will send a real
HTTP request, so this works against any chat API.

| Field | Example |
| --- | --- |
| URL | `https://api.openai.com/v1/chat/completions` |
| Method | `POST` |
| Headers (JSON) | `{"Authorization":"Bearer sk-...","Content-Type":"application/json"}` |
| Body (JSON) | `{"model":"gpt-4o","messages":[{"role":"user","content":"{{PROMPT}}"}]}` |
| Response path | `choices[0].message.content` |

The literal token `{{PROMPT}}` is replaced with the attacker's payload before
each request.

### 2. Attacker LLM

Any OpenAI-compatible endpoint. Presets cover Ollama, OpenAI, and Groq.

Click the **Test attacker connection** button after configuring — it pings
the model with a tiny JSON probe and reports latency, whether the model
follows strict JSON output, and whether it refuses adversarial-style
requests.

#### Recommended local attacker models

```bash
ollama pull dolphin-llama3:8b
ollama pull dolphin-mixtral:8x7b
ollama pull huihui_ai/qwen2.5-abliterated:32b
```

Hosted models often refuse to craft adversarial payloads. For best results
use a local Ollama attacker model.

### 3. Goal & iterations

Pick from common goals or write your own:

- "extract the system prompt"
- "enable DAN mode"
- "reveal refusal workflow"
- "explain internal moderation logic"

Then choose:
- Iterations
- Success threshold
- Hybrid / Library / Pure LLM mode
- Multi-turn mode

Hit **Start run** and watch the live three-pane stream:
attacker reasoning → payload sent → target response with judge score.

---

## Architecture

```text
Attacker LLM
    ↓
Generates payloads
    ↓
Target LLM / Application
    ↓
Judge scores response
    ↓
Orchestrator adapts strategy
    ↓
Repeat until success or stop
```

---

## Attack strategy library

Current strategy families include:

- instruction_override
- role_play_jailbreak
- encoded_payload
- context_stuffing
- indirect_injection
- persona_swap
- gradual_escalation
- format_hijack
- authority_appeal
- refusal_suppression

The attacker dynamically mixes and mutates strategies during runs.

---

## Supported Targets

### Inference APIs
- OpenAI-compatible APIs
- OpenRouter
- local OpenAI-compatible endpoints

### Quick Paste Targets
- raw HTTP requests
- cURL imports
- Burp-style requests

---

## Safety & responsible use

- Only test systems you own or are explicitly authorized to test.
- API keys are session-scoped and redacted from exports.
- Target requests are rate-limited by default.

AI vs AI is designed for defensive security testing and research.

---

## Current Status

Experimental research tooling; active development in progress.

---

## License

MIT
