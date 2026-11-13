<p align="center">
  <img src="docs/assets/condrox-logo.png" alt="Condrox AI" width="180"/>
</p>

<h1 align="center">Condrox AI</h1>

<p align="center">
  <b>A deterministic cognitive conversational architecture — built from scratch by one person.</b><br/>
  It processes language, retrieves knowledge, maintains conversational memory, performs explicit reasoning,
  and evaluates its own responses — without relying on large language models.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/tests-930%20passing-brightgreen" alt="Tests: 930 passing"/>
  <img src="https://img.shields.io/badge/LLM-none-orange" alt="No LLM"/>
</p>

---

## What is Condrox?

> **Early-stage research release** — this is an inspectable foundation, not a
> finished product. Rough edges are documented, not hidden.

Condrox is a working conversational intelligence system that produces coherent,
context-aware, emotionally-aware responses **without any large language model**.
There is no external API call, no model weights, no probabilistic text
generator. Every response is produced by a deterministic 18-stage cognitive
pipeline — and every stage is readable Python you can inspect, trace, and verify.

It exists to prove a point: a carefully engineered symbolic/hybrid architecture
can hold a real conversation — and explain how it arrived at every answer.

**Designed and built from scratch by [Tobias Østen](AUTHORS) (Norway).**

### Live output (real, unedited)

```bash
$ curl -X POST http://localhost:8000/chat \
    -H "X-API-Key: $CONDROX_API_KEY" -H "Content-Type: application/json" \
    -d '{"message": "Hello, what are you?"}'
```

```json
{
  "response": "I am Condrox AI — a cognitive intelligence platform that understands
   language without large language models. ...",
  "intent": "identity", "domain": "condrox", "mode": "explain",
  "emotion": "neutral", "confidence": 0.80,
  "response_time_ms": 14.9, "errors": []
}
```

Answers in **~15 ms**, with intent, domain, emotion, and confidence — because
nothing is predicted, everything is computed.

### More real interactions (unedited)

**Code generation** — intent routing → code mode → working output:

```json
Q: "write a python function to check if a number is prime"
{
  "response": "```python\ndef is_prime(n: int) -> bool:\n    \"\"\"Check if a number is
   prime. Complexity: T(n) ∈ O(√n), S(n) ∈ O(1).\"\"\"\n    if n < 2: return False
   if n == 2: return True\n    if n % 2 == 0: return False\n    ...",
  "intent": "code", "mode": "code", "confidence": 0.88
}
```

**Evaluating a false claim** — recognizes `2+2=5` as a known logical falsehood
rather than agreeing:

> *"2 + 2 = 5 or two plus two equals five is a mathematical falsehood which is
> used as an example of a simple logical error that's obvious to anyone familiar
> with basic arithmetic."*

**Graceful abstention** — instead of confabulating when knowledge is missing:

> *"I don't have specific knowledge about that in my current knowledge base.
> I'm a non-LLM AI, so I can only answer from what I've been taught through my
> knowledge backend."*

**Emotional routing** — `"I'm feeling sad today"` → `intent: emotional`,
`mode: personal`, and a supportive response — no template "sorry" wrapper,
the emotional-detection stack actually gates the reply.

---

## Why it matters

| Property | Typical LLM chatbot | Condrox |
|---|---|---|
| Output | Token prediction (probabilistic) | Deterministic pipeline (same state → same output) |
| Internals | Opaque weights | Every stage inspectable in source |
| Factual generation | Probabilistic — can confabulate fluently | Grounded in retrieved knowledge + explicit reasoning; abstains gracefully when it lacks data. **Not immune to wrong answers** — faulty retrieval or rules can still misfire |
| Traceability | Post-hoc interpretation | Reasoning trace per stage |
| Memory | Context window | Persistent semantic + conversational memory graph |
| Dependencies | Model weights (GBs) | Pure Python + NumPy |

## Architecture

Requests flow through a single orchestrated pipeline. Each numbered stage is a
plain Python module under `core/pipeline/` — read them in order to understand
exactly how a message becomes a response.

```
input ──► ┌─ 01  preprocessor            normalize, tokenize, sanitize
          ├─ 02  intent                  Word Identity Engine → intent class
          ├─ 02b intent enrichment       multi-word phrase fallback
          ├─ 03  mode selector           default / code / analyze / architect / explain
          ├─ 04  domain router           knowledge domain classification
          ├─ 05  context builder         session context assembly
          ├─ 05b context enrichment
          ├─ 06  knowledge fetch         BM25 + Jaccard-graph expansion + embeddings
          ├─ 07  memory fetch            conversational memory graph (neighbor retrieval)
          ├─ 07b memory store            write-back of new memory
          ├─ 08  reasoning core          deterministic reasoning facts
          ├─ 08b agent execution         specialist mode agents
          ├─ 09  synthesis               response composition
          ├─ 09b response quality        quality guardrail gate
          ├─ 09c pattern learning        user pattern learner
          ├─ 09d metacognition           self-check on output
          ├─ 10  personality             tone/style application
          └─ 11  ethics                  final content check
                                       └──► response
```

### What's inside `core/`

| Subsystem | Contents |
|---|---|
| `pipeline/` | The 18-stage deterministic pipeline (above) |
| `knowledge/` | BM25 backend, knowledge router, **knowledge graph** (Jaccard token-overlap edges for query expansion) |
| `memory/` | Semantic memory + **conversational memory graph** (weighted linked nodes, neighbor retrieval) |
| `conversational/` | 26 conversational-intelligence components |
| `agents/` | Mode agents (default, code, analyze, architect, explain) + adapters |
| `word_identity/` | Word Identity Engine — O(1) lookup over modular JSON word stores |
| `personality/` | Profile-based tone and response styling |
| `creative/` | Emotion → music-parameter generation stack |
| `os/` | AI-OS: EventBus, ServiceManager, BootManager, MonitorLoop, StateManager, TaskQueue, SelfRepairManager, kernel |
| `cognition/` | Context manager, strategy manager, meta-learning, pattern learning |
| `security/` | Input sanitization + output PII/secret checks |
| `ethics/` | Content policy evaluation |
| `quality/` | Quality gates on generated output |
| `storage/` | SQLite persistence backend |

Top-level `agents/` additionally contains standalone specialist agents:
`DeveloperAgent` (code analysis/generation engine), `AdminAgent`,
`PersonalAgent`, and `PreReleaseAgent` (a thin public-facing API wrapper with
its own stricter hardening — trusted hosts, body-size limits, IP banning).

### Honest terminology

- **Deterministic** — the pipeline has no sampling, no RNG in response
  selection; identical state produces identical output.
- **Hybrid retrieval** — symbolic rules + BM25 keyword scoring + a
  Jaccard-similarity knowledge graph used to *expand* retrieval, plus
  embedding caches where configured.
- **Graph structures, not a neural brain** — Condrox uses lightweight
  in-process graphs (knowledge graph, memory graph) as data structures for
  retrieval. It does **not** contain a neural network, a weight store, or a
  large engineered brain graph.
- **LLM-free** — verified: `core/`, `api/`, and `agents/` contain zero imports
  of OpenAI, Anthropic, transformers, PyTorch, TensorFlow, llama.cpp, or
  Ollama. `requirements.txt` has no model dependencies.

## Quickstart

Requirements: **Python 3.11+**, ~2 GB disk for a venv. Windows, Linux, or macOS.

```bash
git clone <repo-url> condrox && cd condrox

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Configure (optional — all values have defaults):

```bash
cp .env.example .env      # then edit, or export variables directly
export CONDROX_API_KEY="pick-a-long-random-key"
export CONDROX_SKIP_KNOWLEDGE_LOAD=1   # fastest first boot
```

Run:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
# {"status":"healthy","score":1.0}

curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: pick-a-long-random-key" \
  -H "Content-Type: application/json" \
  -d '{"message":"what is a hash table?"}'
```

For auth-free local hacking: `export CONDROX_AUTH_ENABLED=0` (never in production).

### Knowledge data (optional)

Condrox boots and converses with an empty knowledge backend — answers come from
its self-knowledge pack, reasoning, and agents. To give it broad factual
knowledge, download the public datasets:

```bash
python scripts/download_official_datasets.py --source gsm8k   # single source
python scripts/download_official_datasets.py                  # all sources
```

Chunks land in `data/knowledge/` and are loaded at boot (unless
`CONDROX_SKIP_KNOWLEDGE_LOAD=1`). All sources are free, no API keys.

## API

Main service (`api/main.py`) — auth via `X-API-Key` header or `/auth/login`
session token; rate-limited; CORS-restricted.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + health score |
| `GET /status` | Detailed runtime status |
| `POST /chat` | Conversational endpoint (intent, emotion, confidence in response) |
| `POST /process` | Raw pipeline processing |
| `POST /process/stream` | Streaming processing |
| `POST /creative` | Emotion→music creative stack |
| `GET /metrics` | Prometheus metrics |
| `GET /aios/*` | AI-OS subsystem inspection (services, events, task queue, data flow, health, shutdown) |
| `GET /monitoring/*` | Live interaction monitor + stats |
| `POST /auth/*` | Login / logout / verify |
| `/admin/*` | Admin API (memory/knowledge/agent/pipeline CRUD — API-key required) |

## Testing

```bash
python -m pytest tests/ -q
# 930 tests — the suite asserts real algorithmic behavior,
# not mocks. Runtime: ~5 min.
```

The suite verifies component correctness (pipeline stages, retrieval scoring,
memory graphs, gates, agents, storage, API). It does **not** by itself prove
conversational quality — for that, see `verification/scenarios/` (behavioral
scenarios) and the live demo examples above.

## Security notes

- Always set `CONDROX_API_KEY` for any non-localhost deployment.
- `CONDROX_AUTH_ENABLED=0` is for local development only.
- The API is rate-limited per-route; tighten via `CONDROX_RATE_LIMIT*` env vars.
- Input is sanitized and output is scanned for PII/secret leakage.
- Never commit `.env`, databases, or `data/` — see `.gitignore`.

## Limitations

- Knowledge is bounded by what you download — with an empty backend, factual
  answers degrade gracefully to "I don't know" rather than fabricating.
- Retrieval-quality answers reflect the datasets chosen; it will not improvise
  facts it does not hold.
- **Retrieval can misfire** — lexical similarity sometimes surfaces loosely
  related content (e.g. "hash table" → periodic table). Deterministic is not
  the same as always-right.
- **Session memory is limited** — it stores per-session context, but
  user-specific recall across a conversation is shallow.
- English-centric word-identity stores.
- The specialist agents cover their declared domains; Condrox is not a
  general autonomous agent.

## Project layout

```
condrox/
├── api/               FastAPI service (main + admin routes)
├── core/              The cognitive architecture (pipeline, knowledge, memory, …)
├── agents/            Standalone specialist agents
├── config/            system_config.yaml + dataset sources
├── tests/             930-test suite
├── scripts/           Dataset download, ingestion, benchmarks, utilities
├── verification/      JSON scenario definitions for behavior verification
├── app/               Electron desktop app (boots the brain + chat UI)
├── docs/              Architecture, API, configuration, security
├── k8s/               Kubernetes manifests (optional deployment)
├── monitoring/        Prometheus scrape config
├── requirements.txt
└── .env.example
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline stages, subsystems, graph structures, determinism
- [`docs/API.md`](docs/API.md) — full endpoint reference
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — every environment variable
- [`docs/SECURITY.md`](docs/SECURITY.md) — security model and operator rules

## Desktop app (Electron)

A minimal desktop app boots the backend and opens a chat window — no terminal
required. Requires Node.js 18+ **and** the Python quickstart above completed.

```bash
cd app
npm install
npm start
```

The app spawns `uvicorn api.main:app` on `127.0.0.1:8765` with a per-session
API key, waits for `/health`, then opens the chat UI. It uses the repo's
`.venv` Python if present, else system Python. Set `CONDROX_PORT` /
`CONDROX_PYTHON` env vars to override.

## Author

Condrox AI was designed and built from scratch by **Tobias Østen** (Norway) —
see [`AUTHORS`](AUTHORS). Contact: support@novaspire-ai.com

## License

Apache License 2.0 — Copyright © 2026 Tobias Østen. See `LICENSE`.
