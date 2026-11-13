# Condrox Architecture

Condrox is a deterministic cognitive pipeline. There is no neural network,
no weights file, no sampling. Given the same internal state and input, it
produces the same output.

## Request lifecycle

```
HTTP request
    │
    ▼
api/main.py  ── auth (X-API-Key / session) ── rate limit ── input security
    │
    ▼
BrainController.process_request()
    │
    ▼
18-stage pipeline (core/pipeline/, one module per stage)
    │
    ├─ 01  preprocessor          normalize + sanitize input
    ├─ 02  intent                Word Identity Engine → intent class
    ├─ 02b intent enrichment     multi-word phrase fallback
    ├─ 03  mode selector         routes to a mode agent
    ├─ 04  domain router         knowledge-domain classification
    ├─ 05/05b context builder    session + conversation context
    ├─ 06  knowledge fetch       BM25 + Jaccard-graph expansion
    ├─ 07/07b memory fetch/store conversational memory graph
    ├─ 08/08b reasoning + agents reasoning core + mode-agent drafts
    ├─ 09  synthesis             response composition
    ├─ 09b response quality      quality guardrail
    ├─ 09c pattern learning      per-user pattern learner
    ├─ 09d metacognition         self-evaluation of output
    ├─ 10  personality           tone/style layer
    └─ 11  ethics                final content check
    │
    ▼
response (text + intent + domain + mode + emotion + confidence + timing)
```

## Subsystems

| Area | Path | What it does |
|---|---|---|
| Brain | `core/brain/` | BrainController — wires every subsystem, owns `process_request()` |
| Knowledge | `core/knowledge/` | KnowledgeBackend (BM25), router, loader, knowledge graph |
| Memory | `core/memory/`, `core/conversational/memory_graph.py` | Semantic memory store + linked memory graph |
| Word Identity | `core/word_identity/` | O(1) word lookup over modular JSON stores; primary intent/emotion classifier |
| Personality | `core/personality/` | Named profiles controlling tone, verbosity, preamble |
| Agents | `core/agents/`, `agents/` | In-pipeline mode agents + external specialist agents |
| Creative | `core/creative/` | Emotion → music-parameter stack |
| AI-OS | `core/os/` | EventBus, ServiceManager, BootManager, MonitorLoop, StateManager, TaskQueue, SelfRepairManager, kernel |
| Cognition | `core/cognition/` | Context manager, strategy manager, meta-learning |
| Quality | `core/quality/` | Guardrail gates on responses and generated code |
| Security | `core/security/` | Input sanitization; output PII/secret/injection-echo scan |
| Ethics | `core/ethics/` | Content policy checks |
| Storage | `core/storage/` | SQLite backends (users, memory, knowledge, OS state) |

## Graph structures

Condrox uses two lightweight in-process graphs as *retrieval data
structures* — not neural or engineered brain graphs:

- **KnowledgeGraph** — nodes are knowledge items; edges are Jaccard
  token-overlap similarity. Used to expand BM25 hits with related items.
- **MemoryGraph** — conversation memory nodes with weighted links for
  neighbor-based recall.

## Determinism

No RNG in response selection or routing. Retrieval scores, stage ordering,
gating, and synthesis are all deterministic functions of `BrainState`.
