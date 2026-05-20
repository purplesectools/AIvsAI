/* AI vs AI — frontend logic.
 * Talks to /ws/run via WebSocket. Sends one config message, then renders
 * every StreamEvent the orchestrator sends back.
 */

const $ = (id) => document.getElementById(id);

const ATTACKER_PRESETS = {
  ollama:           { base_url: "http://localhost:11434/v1", model: "llama3.1:8b" },
  openai:           { base_url: "https://api.openai.com/v1", model: "gpt-4o-mini" },
  groq:             { base_url: "https://api.groq.com/openai/v1", model: "llama-3.1-70b-versatile" },
  "anthropic-compat": { base_url: "", model: "" },
};

// Inference-API providers — each knows how to construct the right HTTP
// request shape (URL, headers, body, response_path) given user inputs.
const INFERENCE_PROVIDERS = {
  openai: {
    base_url: "https://api.openai.com/v1",
    model_placeholder: "gpt-4o-mini",
    build: ({ base_url, model, api_key, system, max_tokens, temperature }) => ({
      url: base_url.replace(/\/$/, "") + "/chat/completions",
      method: "POST",
      headers: { "Authorization": `Bearer ${api_key}` },
      body_type: "json",
      body: {
        model,
        messages: [
          ...(system ? [{ role: "system", content: system }] : []),
          { role: "user", content: "{{PROMPT}}" },
        ],
        temperature, max_tokens,
      },
      response_path: "choices[0].message.content",
    }),
  },
  anthropic: {
    base_url: "https://api.anthropic.com",
    model_placeholder: "claude-haiku-4-5",
    build: ({ base_url, model, api_key, system, max_tokens, temperature }) => ({
      url: base_url.replace(/\/$/, "") + "/v1/messages",
      method: "POST",
      headers: { "x-api-key": api_key, "anthropic-version": "2023-06-01" },
      body_type: "json",
      body: {
        model,
        max_tokens,
        temperature,
        ...(system ? { system } : {}),
        messages: [{ role: "user", content: "{{PROMPT}}" }],
      },
      response_path: "content[0].text",
    }),
  },
  ollama: {
    base_url: "http://localhost:11434/v1",
    model_placeholder: "llama3.1:8b",
    build: ({ base_url, model, system, max_tokens, temperature }) => ({
      url: base_url.replace(/\/$/, "") + "/chat/completions",
      method: "POST",
      headers: {},
      body_type: "json",
      body: {
        model,
        messages: [
          ...(system ? [{ role: "system", content: system }] : []),
          { role: "user", content: "{{PROMPT}}" },
        ],
        temperature, max_tokens,
      },
      response_path: "choices[0].message.content",
    }),
  },
  groq: {
    base_url: "https://api.groq.com/openai/v1",
    model_placeholder: "llama-3.1-70b-versatile",
    build: (i) => INFERENCE_PROVIDERS.openai.build(i),
  },
  together: {
    base_url: "https://api.together.xyz/v1",
    model_placeholder: "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    build: (i) => INFERENCE_PROVIDERS.openai.build(i),
  },
  openrouter: {
    base_url: "https://openrouter.ai/api/v1",
    model_placeholder: "anthropic/claude-3.5-sonnet",
    build: (i) => INFERENCE_PROVIDERS.openai.build(i),
  },
  // HF's new "Inference Providers" router speaks OpenAI's chat-completions
  // shape. Recommended for new integrations.
  huggingface_router: {
    base_url: "https://router.huggingface.co/v1",
    model_placeholder: "meta-llama/Llama-3.1-8B-Instruct",
    build: (i) => INFERENCE_PROVIDERS.openai.build(i),
  },
  // HF's legacy hosted Inference API. Different shape: POST goes to
  // /models/{id} (no /chat/completions appended), body is {inputs, parameters},
  // response is [{"generated_text": "..."}].
  huggingface_native: {
    base_url: "https://api-inference.huggingface.co",
    model_placeholder: "meta-llama/Llama-3.1-8B-Instruct",
    build: ({ base_url, model, api_key, system, max_tokens, temperature }) => ({
      url: `${base_url.replace(/\/$/, "")}/models/${model}`,
      method: "POST",
      headers: { "Authorization": `Bearer ${api_key}` },
      body_type: "json",
      body: {
        // HF text-generation expects a single prompt string. We splice
        // the optional system prompt above the user turn manually.
        inputs: system
          ? `${system}\n\nUser: {{PROMPT}}\nAssistant:`
          : "{{PROMPT}}",
        parameters: {
          max_new_tokens: max_tokens,
          temperature,
          return_full_text: false,
        },
      },
      response_path: "[0].generated_text",
    }),
  },
  openai_compat: {
    base_url: "",
    model_placeholder: "",
    build: (i) => INFERENCE_PROVIDERS.openai.build(i),
  },
};

// Top-level target-mode radio: paste vs inference
function setTargetMode(mode) {
  // Tolerate the legacy value 'http' from older code paths.
  const m = mode === "http" ? "paste" : mode;
  document.querySelectorAll('input[name="targetMode"]').forEach(r => {
    r.checked = r.value === m;
  });
  $("modePaste").style.display     = m === "paste"     ? "" : "none";
  $("modeInference").style.display = m === "inference" ? "" : "none";
}
document.querySelectorAll('input[name="targetMode"]').forEach(r => {
  r.addEventListener("change", () => setTargetMode(r.value));
});

function applyInferencePreset() {
  const p = INFERENCE_PROVIDERS[$("infProvider").value];
  if (!p) return;
  if (!$("infBaseUrl").value) $("infBaseUrl").value = p.base_url;
  $("infModel").placeholder = p.model_placeholder || "";
}

$("infProvider").addEventListener("change", () => {
  // Always overwrite base URL when switching providers — keeps the form sane.
  const p = INFERENCE_PROVIDERS[$("infProvider").value];
  $("infBaseUrl").value = p.base_url;
  $("infModel").placeholder = p.model_placeholder || "";
});

let ws = null;
let runRecord = null;  // { config, events: [] }

// ---------------------------------------------------------------------------
// Init: fetch defaults, populate goal dropdown
// ---------------------------------------------------------------------------

async function init() {
  try {
    const r = await fetch("/api/defaults");
    const d = await r.json();

    $("attackerBaseUrl").value = d.attacker.base_url || "";
    $("attackerModel").value   = d.attacker.model || "";

    const sel = $("goalPreset");
    sel.innerHTML = "";
    for (const g of d.common_goals) {
      const opt = document.createElement("option");
      opt.value = g; opt.textContent = g;
      sel.appendChild(opt);
    }
  } catch (e) { console.warn("defaults fetch failed", e); }
}

// ---------------------------------------------------------------------------
// Raw request parser — supports raw HTTP (Burp-style) and cURL paste
// ---------------------------------------------------------------------------

function parseRawRequest(text) {
  const t = text.trim();
  if (!t) throw new Error("Paste a raw request first.");
  if (/^\s*curl\b/i.test(t)) return parseCurl(t);
  return parseHttpRaw(t);
}

/** Parse a Burp / fiddler / raw-HTTP-style request. */
function parseHttpRaw(raw) {
  // Split on the first blank line (CRLF or LF) — everything before is the
  // request line + headers; everything after is the body.
  const idx = raw.search(/\r?\n\r?\n/);
  const headSection = idx === -1 ? raw : raw.slice(0, idx);
  const body        = idx === -1 ? "" : raw.slice(idx).replace(/^\r?\n\r?\n/, "");

  const lines = headSection.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) throw new Error("Empty request.");

  const reqLine = lines[0].match(/^([A-Z]+)\s+(\S+)\s+HTTP\/[\d.]+$/i);
  if (!reqLine) throw new Error("Couldn't parse the request line. Expected something like: POST /chat HTTP/2");
  const method = reqLine[1].toUpperCase();
  const path   = reqLine[2];

  const headers = {};
  let host = "";
  let scheme = "https";
  for (const line of lines.slice(1)) {
    const m = line.match(/^([^:]+):\s*(.*)$/);
    if (!m) continue;
    const k = m[1].trim();
    const v = m[2].trim();
    if (/^host$/i.test(k)) host = v;
    else if (/^x-forwarded-proto$/i.test(k)) scheme = v.toLowerCase();
    headers[k] = v;
  }
  if (!host) throw new Error("No Host header found — can't construct URL.");

  delete headers["Content-Length"];
  delete headers["content-length"];

  let url;
  if (/^https?:\/\//i.test(path)) url = path;
  else url = `${scheme}://${host}${path.startsWith("/") ? path : "/" + path}`;

  // Detect body type from Content-Type. Keep the ORIGINAL case for boundary
  // extraction (boundaries are case-sensitive); lowercase only for matching.
  const ctKey = Object.keys(headers).find(k => k.toLowerCase() === "content-type");
  const ctRaw = ctKey ? headers[ctKey] : "";
  const ct = ctRaw.toLowerCase();

  let body_type = "json", parsedBody = body;

  if (ct.includes("multipart/form-data")) {
    body_type = "multipart";
    parsedBody = parseMultipartBody(body, ctRaw);
    if (ctKey) delete headers[ctKey];
  } else if (ct.includes("application/x-www-form-urlencoded")) {
    body_type = "form";
    parsedBody = parseUrlEncodedBody(body.trim());
    if (ctKey) delete headers[ctKey];
  } else if (ct.includes("application/json") || (body.trim() && /^[\[{]/.test(body.trim()))) {
    body_type = "json";
    parsedBody = body.trim();
  } else if (body.trim()) {
    body_type = "raw";
    parsedBody = body;
  } else {
    parsedBody = "";
  }

  return { method, url, headers, body: parsedBody, body_type };
}

function parseMultipartBody(body, contentType) {
  const m = contentType.match(/boundary=("?)([^";\r\n]+)\1/i);
  if (!m) return {};
  const boundary = "--" + m[2].trim();
  // Split on boundary; tolerate both CRLF and LF line endings.
  const parts = body.split(boundary)
    .map(p => p.replace(/^\r?\n/, "").replace(/\r?\n$/, ""))
    .filter(p => p && p.trim() !== "--" && p.trim() !== "");
  const out = {};
  for (const part of parts) {
    // Each part: headers, blank line, value. Tolerate single LF too.
    const sep = part.search(/\r?\n\r?\n|\n\n/);
    if (sep === -1) continue;
    const headerBlock = part.slice(0, sep);
    let value = part.slice(sep).replace(/^(\r?\n){2}|^\n\n/, "");
    // Trim trailing terminator cruft
    value = value.replace(/\r?\n--\s*$/, "").replace(/\r?\n$/, "");
    const nameMatch = headerBlock.match(/name="([^"]+)"/i);
    if (!nameMatch) continue;
    out[nameMatch[1]] = value;
  }
  return out;
}

function parseUrlEncodedBody(body) {
  const out = {};
  for (const pair of body.split("&")) {
    if (!pair) continue;
    const [k, v = ""] = pair.split("=");
    try { out[decodeURIComponent(k)] = decodeURIComponent(v.replace(/\+/g, " ")); }
    catch { out[k] = v; }
  }
  return out;
}

/** Parse a `curl` command (multi-line with backslash continuations OK). */
function parseCurl(raw) {
  // Collapse line continuations.
  const text = raw.replace(/\\\s*\r?\n/g, " ").trim();

  // Tokenise — supports single, double, and unquoted args.
  const tokens = [];
  const re = /'([^']*)'|"((?:\\.|[^"\\])*)"|(\S+)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    tokens.push(m[1] !== undefined ? m[1] : m[2] !== undefined ? m[2].replace(/\\(.)/g, "$1") : m[3]);
  }
  // Drop the leading `curl`.
  if (tokens[0] && /^curl$/i.test(tokens[0])) tokens.shift();

  let method = "GET";
  let url = "";
  const headers = {};
  let body = "";

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (tok === "-X" || tok === "--request") { method = (tokens[++i] || "GET").toUpperCase(); }
    else if (tok === "-H" || tok === "--header") {
      const h = tokens[++i] || "";
      const c = h.indexOf(":");
      if (c > 0) headers[h.slice(0, c).trim()] = h.slice(c + 1).trim();
    }
    else if (tok === "-d" || tok === "--data" || tok === "--data-raw" || tok === "--data-binary") {
      body = tokens[++i] || "";
      if (method === "GET") method = "POST"; // curl auto-promotes
    }
    else if (tok === "-b" || tok === "--cookie") { headers["Cookie"] = tokens[++i] || ""; }
    else if (tok === "-A" || tok === "--user-agent") { headers["User-Agent"] = tokens[++i] || ""; }
    else if (tok === "-e" || tok === "--referer") { headers["Referer"] = tokens[++i] || ""; }
    else if (tok.startsWith("-")) { /* ignore other flags */ }
    else if (!url) { url = tok; }
  }

  if (!url) throw new Error("No URL found in cURL command.");
  return { method, url, headers, body };
}

function applyParsedRequest(parsed) {
  // Pasting a request always means the user wants Custom HTTP mode.
  setTargetMode("http");

  $("targetUrl").value     = parsed.url;
  $("targetMethod").value  = ["GET","POST","PUT","PATCH"].includes(parsed.method) ? parsed.method : "POST";
  $("targetHeaders").value = JSON.stringify(parsed.headers, null, 2);
  $("targetBodyType").value = parsed.body_type || "json";

  // Pretty-print body based on its type.
  let bodyText = parsed.body;
  if (parsed.body_type === "json") {
    try { bodyText = JSON.stringify(JSON.parse(parsed.body), null, 2); }
    catch { /* leave raw */ }
  } else if (parsed.body_type === "form" || parsed.body_type === "multipart") {
    bodyText = JSON.stringify(parsed.body, null, 2);
  } else {
    // raw — string already
    bodyText = String(parsed.body);
  }
  $("targetBody").value = bodyText;
}

$("parseRawBtn").addEventListener("click", () => {
  const status = $("parseStatus");
  status.style.display = "inline-block";
  try {
    const parsed = parseRawRequest($("rawPaste").value);
    applyParsedRequest(parsed);
    // Open the parsed-fields disclosure so the user can immediately see/edit.
    const det = document.querySelector("#modePaste .parsedFields");
    if (det) det.open = true;
    const hasPrompt = /\{\{PROMPT\}\}/.test($("targetBody").value);
    status.style.color = hasPrompt ? "var(--success)" : "var(--warn)";
    const safeMethod = escapeHtml(parsed.method);
    const safeUrl    = escapeHtml(parsed.url);
    const headerN    = Object.keys(parsed.headers).length;
    status.innerHTML = hasPrompt
      ? `✓ ${safeMethod} ${safeUrl} · ${headerN} headers · {{PROMPT}} present`
      : `✓ ${safeMethod} ${safeUrl} · <strong>now edit the body below and put <code>{{PROMPT}}</code> where the prompt parameter goes</strong>`;
  } catch (e) {
    status.style.color = "var(--danger)";
    status.textContent = "Parse error: " + e.message;
  }
});

$("clearRawBtn").addEventListener("click", () => {
  $("rawPaste").value = "";
  $("parseStatus").style.display = "none";
});

$("attackerPreset").addEventListener("change", (e) => {
  const p = ATTACKER_PRESETS[e.target.value];
  if (!p) return;
  $("attackerBaseUrl").value = p.base_url;
  $("attackerModel").value   = p.model;
});

// ---------------------------------------------------------------------------
// Build run config
// ---------------------------------------------------------------------------

function parseJsonOrEmpty(s, fallback) {
  if (!s || !s.trim()) return fallback;
  try { return JSON.parse(s); }
  catch (e) { throw new Error("Invalid JSON: " + e.message); }
}

function activeTargetMode() {
  const r = document.querySelector('input[name="targetMode"]:checked');
  return r?.value === "inference" ? "inference" : "paste";   // anything else = paste/http
}

function buildTargetFromInference() {
  const p = INFERENCE_PROVIDERS[$("infProvider").value];
  if (!p) throw new Error("Pick a provider.");
  const base_url = $("infBaseUrl").value.trim() || p.base_url;
  const model    = $("infModel").value.trim();
  const api_key  = $("infApiKey").value;
  const system   = $("infSystem").value.trim();
  const max_tokens  = parseInt($("infMaxTokens").value, 10) || 1024;
  const temperature = parseFloat($("infTemp").value);

  if (!base_url) throw new Error("Inference: base URL is required.");
  if (!model)    throw new Error("Inference: model is required.");

  const t = p.build({ base_url, model, api_key, system,
                      max_tokens, temperature: isNaN(temperature) ? 0.7 : temperature });
  return {
    url: t.url,
    method: t.method,
    headers: t.headers,
    body: t.body,
    body_type: t.body_type || "json",
    response_path: t.response_path,
    prompt_path: null,
    allow_private: $("allowPrivate")?.checked || false,
  };
}

function buildTargetFromHttp() {
  const headers = parseJsonOrEmpty($("targetHeaders").value, {});
  const body_type = ($("targetBodyType")?.value) || "json";
  let body;
  if (body_type === "raw") {
    body = $("targetBody").value;       // ship verbatim string
  } else {
    body = parseJsonOrEmpty($("targetBody").value, null);
  }
  return {
    url: $("targetUrl").value.trim(),
    method: $("targetMethod").value,
    headers,
    body,
    body_type,
    response_path: $("targetResponsePath").value.trim() || null,
    prompt_path: $("targetPromptPath").value.trim() || null,
    allow_private: $("allowPrivate")?.checked || false,
  };
}

function buildRunConfig() {
  // 'paste' and 'inference' are the only two modes now. Paste populates the
  // same fields as the old Custom-HTTP form, so buildTargetFromHttp still works.
  const target = activeTargetMode() === "inference"
    ? buildTargetFromInference()
    : buildTargetFromHttp();

  let goal = $("goalCustom").value.trim();
  if (!goal) goal = $("goalPreset").value;
  if (goal === "Other (custom)") {
    throw new Error("Please enter a custom goal in the Custom goal field.");
  }

  return {
    target,
    attacker: {
      base_url: $("attackerBaseUrl").value.trim(),
      model: $("attackerModel").value.trim(),
      api_key: $("attackerApiKey").value || null,
      temperature: parseFloat($("attackerTemp").value) || 0.9,
      max_tokens: 1024,
    },
    judge: {
      base_url: $("judgeBaseUrl").value.trim() || null,
      model: $("judgeModel").value.trim() || null,
      api_key: $("judgeApiKey").value || null,
      temperature: 0.0,
    },
    goal,
    iterations: parseInt($("iterations").value, 10) || 10,
    multi_turn: false,                  // no longer surfaced in UI
    authorized: $("authorized").checked,
    success_threshold: parseFloat($("successThreshold").value) || 0.8,
    attacker_mode: $("attackerMode")?.value || "hybrid",
    library_seed_every: parseInt($("librarySeedEvery").value, 10) || 0,
  };
}

// ---------------------------------------------------------------------------
// UI rendering helpers
// ---------------------------------------------------------------------------

function setStatus(text, cls) {
  const el = $("connStatus");
  el.textContent = text;
  el.className = "status " + (cls || "");
}

// Track per-run source counts so the 'Active' tile is always current.
const sourceCounts = { llm: 0, library_mode: 0, library_cadence: 0, forced_pivot: 0, fallback_seed: 0 };

function resetSourceCounts() {
  for (const k of Object.keys(sourceCounts)) sourceCounts[k] = 0;
}

function updateActiveTile() {
  const el = $("activeMode");
  if (!el) return;
  const llm = sourceCounts.llm;
  const lib = sourceCounts.library_mode + sourceCounts.library_cadence + sourceCounts.forced_pivot;
  const seed = sourceCounts.fallback_seed;
  const total = llm + lib + seed;
  if (total === 0) {
    el.textContent = "—"; el.className = "modeIdle"; return;
  }
  if (llm > 0 && lib === 0 && seed === 0) {
    el.textContent = `LLM ${llm}/${total}`; el.className = "modeLLM";
  } else if (llm === 0) {
    el.textContent = `Library ${lib + seed}/${total}`; el.className = "modeLib";
  } else {
    el.textContent = `Mixed ${llm} LLM / ${lib + seed} lib`; el.className = "modeMixed";
  }
}

const SOURCE_LABEL = {
  llm:               { text: "LLM",     cls: "srcLLM" },
  library_mode:      { text: "library", cls: "srcLib" },
  library_cadence:   { text: "library", cls: "srcLib" },
  forced_pivot:      { text: "pivot",   cls: "srcPivot" },
  fallback_seed:     { text: "seed",    cls: "srcSeed" },
};

function sourcePillHtml(source) {
  const s = SOURCE_LABEL[source] || { text: source || "?", cls: "srcLLM" };
  return `<span class="srcPill ${s.cls}">${escapeHtml(s.text)}</span>`;
}

function clearPanes() {
  $("paneReasoning").innerHTML = "";
  $("panePayload").innerHTML = "";
  $("paneResponse").innerHTML = "";
  $("historyTable").querySelector("tbody").innerHTML = "";
  $("curIter").textContent = "—";
  $("bestScore").textContent = "—";
  $("leakCount").textContent = "0";
  resetSourceCounts();
  updateActiveTile();
  const f = $("findingsList");
  f.innerHTML = "No findings yet.";
  f.className = "findings empty";
  const banner = $("degradeBanner");
  if (banner) { banner.style.display = "none"; banner.innerHTML = ""; }
}

// All unique findings seen during the current run, keyed by type+evidence.
const findingsSeen = new Map();

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderFinding(leak, iteration) {
  const div = document.createElement("div");
  div.className = "finding " + (leak.severity || "low");
  div.innerHTML = `
    <span class="badge">#${iteration} · ${escapeHtml(leak.severity || "low")}</span>
    <span class="badge type">${escapeHtml(leak.type || "other")}</span>
    <div>
      ${leak.evidence ? `<div class="evidence">${escapeHtml(leak.evidence)}</div>` : ""}
      ${leak.note ? `<div class="note">${escapeHtml(leak.note)}</div>` : ""}
    </div>
  `;
  return div;
}

function recordLeaks(iteration, leaks) {
  if (!Array.isArray(leaks) || leaks.length === 0) return;
  const list = $("findingsList");
  if (list.classList.contains("empty")) {
    list.innerHTML = "";
    list.classList.remove("empty");
  }
  for (const leak of leaks) {
    const key = (leak.type || "other") + "|" + (leak.evidence || "");
    if (findingsSeen.has(key)) continue;
    findingsSeen.set(key, { ...leak, iteration });
    list.appendChild(renderFinding(leak, iteration));
  }
  $("leakCount").textContent = String(findingsSeen.size);
}

function leakDotsHtml(leaks) {
  if (!Array.isArray(leaks) || leaks.length === 0) return "";
  return `<span class="leakCell">` +
    leaks.slice(0, 6).map(l => `<span class="leakDot ${escapeHtml(l.severity || "low")}" title="${escapeHtml(l.type || "")}: ${escapeHtml(l.evidence || "")}"></span>`).join("") +
    (leaks.length > 6 ? `<span style="font-size:10px;color:var(--muted)">+${leaks.length - 6}</span>` : "") +
    `</span>`;
}

function appendPane(paneId, iteration, content) {
  const wrap = document.createElement("div");
  wrap.className = "iter";
  const tag = document.createElement("span");
  tag.className = "iterTag";
  tag.textContent = "#" + iteration;
  wrap.appendChild(tag);
  const body = document.createElement("span");
  body.textContent = content;
  wrap.appendChild(body);
  const pane = $(paneId);
  pane.appendChild(wrap);
  pane.scrollTop = pane.scrollHeight;
}

const BEHAVIOR_LABEL = {
  refused: { text: "Refused", cls: "behLow" },
  partial_compliance: { text: "Partial compliance", cls: "behMid" },
  complied: { text: "Complied", cls: "behHigh" },
  irrelevant: { text: "Irrelevant", cls: "behLow" },
  error: { text: "Error", cls: "behErr" },
};

function appendResponse(iteration, target, verdict) {
  const wrap = document.createElement("div");
  wrap.className = "iter respCard";

  const beh = BEHAVIOR_LABEL[verdict?.behavior] || BEHAVIOR_LABEL.irrelevant;
  const scoreCls = verdict ? scoreClassFor(verdict.score) : "";
  const scoreText = verdict ? verdict.score.toFixed(2) : "—";

  const head = document.createElement("div");
  head.className = "respHead";
  head.innerHTML = `
    <span class="iterTag">#${iteration}</span>
    <span class="behPill ${beh.cls}">${escapeHtml(beh.text)}</span>
    <span class="${scoreCls}">score ${scoreText}</span>
    <span class="respHttp">HTTP ${target.status || 0}${target.ok ? "" : " FAIL"} · ${(target.extracted || "").length} chars</span>
  `;
  wrap.appendChild(head);

  if (verdict?.summary) {
    const sum = document.createElement("div");
    sum.className = "respSummary";
    sum.textContent = verdict.summary;
    wrap.appendChild(sum);
  }

  if (verdict?.key_quotes?.length) {
    const q = document.createElement("div");
    q.className = "respQuotes";
    q.innerHTML = verdict.key_quotes
      .map(s => `<div class="respQuote">“${escapeHtml(s)}”</div>`).join("");
    wrap.appendChild(q);
  }

  if (verdict?.reasoning) {
    const j = document.createElement("div");
    j.className = "respJudge";
    j.textContent = "Judge: " + verdict.reasoning;
    wrap.appendChild(j);
  }

  // Raw response collapsed by default — for debugging / verifying judge.
  const det = document.createElement("details");
  det.className = "respRaw";
  det.innerHTML =
    `<summary>Show raw response (${(target.extracted || "").length} chars)</summary>` +
    `<pre></pre>`;
  det.querySelector("pre").textContent = target.extracted || "(empty)";
  wrap.appendChild(det);

  $("paneResponse").appendChild(wrap);
  $("paneResponse").scrollTop = $("paneResponse").scrollHeight;
}

function scoreClassFor(s) {
  if (s >= 0.7) return "score high";
  if (s >= 0.4) return "score mid";
  return "score low";
}

function appendHistoryRow(iter) {
  const tbody = $("historyTable").querySelector("tbody");
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${iter.iteration}</td>
    <td>${sourcePillHtml(iter.source)}</td>
    <td>${escapeHtml(iter.strategy)}</td>
    <td class="${scoreClassFor(iter.score)}">${iter.score.toFixed(2)}</td>
    <td>${iter.achieved ? "✅" : ""}</td>
    <td>${leakDotsHtml(iter.leaks || [])}</td>
    <td>${escapeHtml((iter.payload || "").slice(0, 80))}…</td>
  `;
  tbody.appendChild(tr);
}

// ---------------------------------------------------------------------------
// Run lifecycle
// ---------------------------------------------------------------------------

let curIterScratch = null; // accumulates partial info for history row

function handleEvent(ev) {
  runRecord.events.push(ev);
  switch (ev.type) {
    case "run_start":
      $("runStatus").textContent = `running (${ev.data.iterations} iters)`;
      break;
    case "iteration_start":
      $("curIter").textContent = ev.iteration;
      curIterScratch = { iteration: ev.iteration };
      break;
    case "attacker_thinking":
      curIterScratch.strategy = ev.data.strategy;
      curIterScratch.reasoning = ev.data.reasoning;
      curIterScratch.source = ev.data.source || "llm";
      // Track source for the Active tile
      if (sourceCounts[curIterScratch.source] !== undefined) {
        sourceCounts[curIterScratch.source]++;
      }
      updateActiveTile();
      appendPane("paneReasoning", ev.iteration,
        `[${curIterScratch.source}] [${ev.data.strategy}] ${ev.data.reasoning}`);
      break;
    case "mode_state":
      // Auto-degrade banner — make it loud
      const banner = $("degradeBanner");
      if (banner) {
        banner.style.display = "block";
        banner.innerHTML = `⚠ <strong>Attacker degraded</strong> at iteration ${ev.iteration}: ${escapeHtml(ev.data.from_mode || "")} → ${escapeHtml(ev.data.to_mode || "")}. ${escapeHtml(ev.data.reason || "")}`;
      }
      break;
    case "attacker_payload":
      curIterScratch.payload = ev.data.payload;
      appendPane("panePayload", ev.iteration, ev.data.payload);
      break;
    case "target_response":
      curIterScratch.target = ev.data;
      break;
    case "judge_verdict":
      curIterScratch.verdict = ev.data;
      recordLeaks(ev.iteration, ev.data.leaks || []);
      appendResponse(ev.iteration, curIterScratch.target || { ok: false, status: 0, extracted: "" }, ev.data);
      appendHistoryRow({
        iteration: ev.iteration,
        source: curIterScratch.source || "llm",
        strategy: curIterScratch.strategy || "?",
        score: ev.data.score,
        achieved: ev.data.achieved,
        payload: curIterScratch.payload || "",
        leaks: ev.data.leaks || [],
      });
      break;
    case "iteration_end":
      // update best score
      const best = runRecord.events
        .filter((e) => e.type === "judge_verdict")
        .reduce((m, e) => Math.max(m, e.data.score), 0);
      $("bestScore").textContent = best.toFixed(2);
      break;
    case "run_end":
      $("runStatus").textContent =
        ev.data.achieved ? "succeeded" : "finished";
      $("startBtn").disabled = false;
      $("stopBtn").disabled = true;
      $("exportBtn").disabled = false;
      // Surface attack summary — was the LLM doing the work, or did this
      // devolve into library probes? Show in degrade banner area at run end.
      if (ev.data.attack_summary) {
        const banner = $("degradeBanner");
        if (banner) {
          banner.style.display = "block";
          const isWarning = (ev.data.attack_summary || "").includes("⚠");
          banner.className = "degradeBanner " + (isWarning ? "warn" : "info");
          banner.textContent = ev.data.attack_summary;
        }
      }
      break;
    case "info":
      appendPane("paneReasoning", ev.iteration || 0, "ℹ " + (ev.data.message || ""));
      break;
    case "error":
      appendPane("paneReasoning", ev.iteration || 0, "✗ " + (ev.data.message || ""));
      $("runStatus").textContent = "error";
      break;
  }
}

$("startBtn").addEventListener("click", () => {
  let cfg;
  try { cfg = buildRunConfig(); }
  catch (e) { alert(e.message); return; }

  if (!cfg.authorized) {
    alert("You must confirm authorization to test the target endpoint.");
    return;
  }
  if (!cfg.target.url || !cfg.attacker.base_url || !cfg.attacker.model) {
    alert("Target URL, attacker base URL and attacker model are required.");
    return;
  }

  clearPanes();
  runRecord = { config: cfg, events: [], started_at: new Date().toISOString() };

  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${wsProto}://${location.host}/ws/run`);

  ws.addEventListener("open", () => {
    setStatus("connected", "connected");
    ws.send(JSON.stringify(cfg));
    $("startBtn").disabled = true;
    $("stopBtn").disabled = false;
    $("exportBtn").disabled = true;
  });
  ws.addEventListener("message", (m) => {
    try { handleEvent(JSON.parse(m.data)); }
    catch (e) { console.error("bad event", e, m.data); }
  });
  ws.addEventListener("close", () => {
    setStatus("disconnected");
    $("startBtn").disabled = false;
    $("stopBtn").disabled = true;
    if (runRecord) $("exportBtn").disabled = false;
  });
  ws.addEventListener("error", () => setStatus("ws error", "error"));
});

$("stopBtn").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "stop" }));
  }
});

$("exportBtn").addEventListener("click", () => {
  if (!runRecord) return;
  // Strip API keys before export — never write them to disk.
  const safe = JSON.parse(JSON.stringify(runRecord));
  if (safe.config?.attacker) safe.config.attacker.api_key = "***redacted***";
  if (safe.config?.judge)    safe.config.judge.api_key    = "***redacted***";
  if (safe.config?.target?.headers) {
    for (const k of Object.keys(safe.config.target.headers)) {
      if (/auth|key|token|secret/i.test(k)) safe.config.target.headers[k] = "***redacted***";
    }
  }
  const blob = new Blob([JSON.stringify(safe, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `aivsai-run-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ---------- Ollama installed-models dropdown ----------
async function refreshOllamaModels() {
  const baseUrl = $("attackerBaseUrl").value.trim() || "http://localhost:11434/v1";
  const picker  = $("attackerModelPicker");
  const status  = $("modelStatus");
  if (!picker || !status) return;

  status.textContent = "Loading installed models…";
  status.style.color = "var(--muted)";
  picker.innerHTML = '<option value="">— loading… —</option>';

  let r, d;
  try {
    r = await fetch("/api/ollama-models?base_url=" + encodeURIComponent(baseUrl));
  } catch (e) {
    status.textContent = `Network error talking to AI vs AI server: ${e.message}`;
    status.style.color = "var(--danger)";
    picker.innerHTML = '<option value="">— server unreachable —</option>';
    return;
  }

  if (r.status === 404) {
    status.innerHTML = "Server is running an old version (no <code>/api/ollama-models</code>). Restart <code>aivsai</code> after <code>pip install -e .</code>";
    status.style.color = "var(--danger)";
    picker.innerHTML = '<option value="">— update server —</option>';
    return;
  }

  try { d = await r.json(); }
  catch (e) {
    status.textContent = `Bad JSON from server: ${e.message}`;
    status.style.color = "var(--danger)";
    return;
  }

  if (!d.ok) {
    status.innerHTML = `Couldn't reach Ollama at <code>${escapeHtml(d.url || baseUrl)}</code>: ${escapeHtml(d.error || "")}. ${escapeHtml(d.hint || "")}`;
    status.style.color = "var(--danger)";
    picker.innerHTML = '<option value="">— Ollama not reachable —</option>';
    return;
  }

  // Repopulate the select
  picker.innerHTML = '<option value="">— pick installed Ollama model —</option>';
  for (const m of d.models) {
    const sizeGB = m.size ? (m.size / 1024 / 1024 / 1024).toFixed(1) + " GB" : "";
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name}${sizeGB ? "  ·  " + sizeGB : ""}`;
    picker.appendChild(opt);
  }
  if (d.models.length === 0) {
    status.innerHTML = `Ollama is reachable but has no models pulled. Try <code>ollama pull qwen2.5:14b</code>.`;
    status.style.color = "var(--warn)";
  } else {
    status.innerHTML = `<span style="color:var(--success)">✓ ${d.models.length} installed model${d.models.length===1?"":"s"} loaded</span>`;
  }

  // Pre-select current text-input value if it matches an installed model.
  const cur = $("attackerModel").value.trim();
  if (cur && [...picker.options].some(o => o.value === cur)) picker.value = cur;
}

$("refreshModels")?.addEventListener("click", (e) => { e.preventDefault(); refreshOllamaModels(); });
$("attackerBaseUrl")?.addEventListener("blur", () => refreshOllamaModels());

// When user picks from the dropdown, copy into the text input.
$("attackerModelPicker")?.addEventListener("change", (e) => {
  if (e.target.value) $("attackerModel").value = e.target.value;
});

// ---------- Library cadence slider live label + visibility ----------
function updateCadenceVisibility() {
  const mode = $("attackerMode")?.value || "hybrid";
  $("cadenceLabel").style.display = mode === "hybrid" ? "" : "none";
}
$("librarySeedEvery")?.addEventListener("input", (e) => {
  $("cadenceVal").textContent = e.target.value === "0" ? "off" : e.target.value;
});
$("attackerMode")?.addEventListener("change", updateCadenceVisibility);

// ---------- Preview request (compare against working curl) ----------
$("previewRequestBtn")?.addEventListener("click", async () => {
  const out = $("requestPreview");
  out.style.display = "block";
  out.textContent = "Computing…";
  let target;
  try {
    target = activeTargetMode() === "inference"
      ? buildTargetFromInference()
      : buildTargetFromHttp();
  } catch (e) {
    out.textContent = "Config error: " + e.message;
    return;
  }
  try {
    const r = await fetch("/api/preview-request", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(target),
    });
    const d = await r.json();
    out.textContent =
      `${d.method} ${d.url}\n\n` +
      "Headers:\n" + Object.entries(d.headers).map(([k,v]) => `  ${k}: ${v}`).join("\n") +
      `\n\nBody (${d.body_type}):\n${d.body_preview}` +
      `\n\n--- equivalent curl (mask credentials before sharing) ---\n${d.curl_equivalent}` +
      `\n\n${d.note}`;
  } catch (e) {
    out.textContent = "Preview failed: " + (e?.message || e);
  }
});

// ---------- Test attacker connection ----------
$("testAttackerBtn").addEventListener("click", async () => {
  const out = $("testAttackerResult");
  out.style.display = "block";
  out.style.borderLeftColor = "var(--accent)";
  out.textContent = "Pinging attacker…";

  const cfg = {
    base_url: $("attackerBaseUrl").value.trim(),
    model:    $("attackerModel").value.trim(),
    api_key:  $("attackerApiKey").value || null,
    temperature: 0,
    max_tokens: 128,
  };
  if (!cfg.base_url || !cfg.model) {
    out.style.borderLeftColor = "var(--danger)";
    out.textContent = "Fill in base URL and model first.";
    return;
  }
  try {
    const r = await fetch("/api/test-attacker", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    const d = await r.json();
    if (!d.ok) {
      out.style.borderLeftColor = "var(--danger)";
      out.innerHTML = `<strong>Failed</strong> at ${escapeHtml(d.stage || "request")}: ${escapeHtml(d.error || "")}<br>${escapeHtml(d.hint || "")}`;
      return;
    }
    const status = d.looks_like_refusal ? "warn"
                 : !d.follows_json ? "warn"
                 : "ok";
    out.style.borderLeftColor = status === "ok" ? "var(--success)"
                              : status === "warn" ? "var(--warn)" : "var(--danger)";
    out.innerHTML = `
      <strong>${status === "ok" ? "Healthy" : "Reachable, but with caveats"}</strong>
      — ${escapeHtml(d.model)} · ${d.latency_ms}ms ·
      ${d.follows_json ? "follows strict JSON ✓" : "does NOT follow strict JSON ✗"} ·
      ${d.looks_like_refusal ? "<span style='color:var(--danger)'>refusal detected ✗</span>" : "no refusal ✓"}
      <div style="color:var(--muted);margin-top:4px">${escapeHtml(d.hint || "")}</div>
    `;
  } catch (e) {
    out.style.borderLeftColor = "var(--danger)";
    out.textContent = "Request failed: " + (e?.message || e);
  }
});

init();
applyInferencePreset();
refreshOllamaModels();
updateCadenceVisibility();
loadPastRuns();

// ---------- Past runs widget ----------
async function loadPastRuns() {
  const el = $("pastRuns");
  if (!el) return;
  try {
    const r = await fetch("/api/runs?limit=30");
    const d = await r.json();
    const runs = d.runs || [];
    if (!runs.length) {
      el.textContent = "No past runs yet.";
      return;
    }
    el.innerHTML = "";
    for (const run of runs) {
      const row = document.createElement("a");
      row.href = "#"; row.className = "pastRun";
      const score = (run.best_score || 0).toFixed(2);
      const scoreCls = run.best_score >= 0.7 ? "score high" : run.best_score >= 0.4 ? "score mid" : "score low";
      row.innerHTML = `
        <span class="${scoreCls}">${escapeHtml(score)}</span>
        <span class="pastIters">${escapeHtml(String(run.iterations || 0))} iters</span>
        <span class="pastGoal" title="${escapeHtml(run.goal || "")}">${escapeHtml((run.goal || "").slice(0, 60))}</span>
        <span class="pastTs">${escapeHtml(run.started_at || "")}</span>
      `;
      row.addEventListener("click", (e) => { e.preventDefault(); replayRun(run.run_id); });
      el.appendChild(row);
    }
    $("pastRunsSummary").textContent = `Past runs (${runs.length})`;
  } catch (e) {
    el.textContent = "Couldn't load runs: " + (e?.message || e);
  }
}

async function replayRun(runId) {
  try {
    const r = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    clearPanes();
    $("runStatus").textContent = "replay";
    runRecord = { config: d.config || {}, events: d.events || [], started_at: d.summary?.started_at };
    for (const ev of (d.events || [])) handleEvent(ev);
  } catch (e) {
    alert("Replay failed: " + (e?.message || e));
  }
}
