# Condrox AI Pre-Release Agent

Public, isolated, no-auth chat agent for the Condrox AI website demo.

## Purpose

Prove the AI actually works — live, public, no sign-up. Visitors to the
website can chat with the full 18-stage cognitive pipeline and get real
responses, with strict isolation so different users never contaminate
each other's conversations.

## Isolation Design

| Layer              | Mechanism                                              |
|--------------------|--------------------------------------------------------|
| Brain instance     | Dedicated BrainController (not shared with main API)   |
| Memory writes      | `store_memory=False` — zero persistent writes          |
| Session state      | In-memory dict, TTL'd at 30 min idle                   |
| History depth      | Last 5 turns only                                      |
| Input cap          | 500 chars                                              |
| Output cap         | 1000 chars                                             |
| Rate limit         | 10 req/min per IP                                      |
| Disk persistence   | None — nothing written to SQLite or files              |

## Architecture

```
Website chat.html
      │
      ▼  POST /chat {message, session_id}
PreReleaseAgent API (FastAPI, port 8002)
      │
      ▼  agent.chat(message, session_id)
PreReleaseAgent
      │  ├── get_or_create_session(session_id)  → ephemeral context
      │  ├── build_context()                    → last 5 turns
      │  └── brain.process_request(store_memory=False)
      │         │
      │         ▼
      │    18-stage cognitive pipeline
      │         │
      │         ▼
      │    BrainState (final_output, intent, domain, mode)
      │
      ▼  cap output at 1000 chars
PreReleaseResult
      │
      ▼
ChatResponse → website
```

## Files

```
PreReleaseAgent/
├── __init__.py
├── README.md                          ← this file
├── condrox_pre_release_agent.py       ← agent class (session mgmt, context, caps)
└── api/
    ├── main.py                        ← FastAPI service (/chat, /health)
    └── README.md                      ← API docs
```

## Run

```bash
# Local testing
python -m uvicorn agents.PreReleaseAgent.api.main:app --host 0.0.0.0 --port 8002

# Or directly
python agents/PreReleaseAgent/api/main.py
```

The server boots the dedicated BrainController at startup (~2-5s), then
serves chat requests with zero persistent memory writes.

## Cloudflare Deployment (Future)

For production, the same code runs behind Cloudflare Tunnel or as a
Cloudflare Worker. The isolation guarantees hold regardless of the
hosting layer because they're enforced in the agent code itself.

Copyright (c) 2026 Tobias Østen. All rights reserved.
