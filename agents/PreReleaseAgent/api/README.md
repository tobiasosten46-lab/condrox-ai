# PreReleaseAgent API

FastAPI service exposing the PreReleaseAgent over HTTP.

## Endpoints

| Method | Path           | Auth | Rate Limit  | Description                     |
|--------|----------------|------|-------------|---------------------------------|
| POST   | `/chat`        | None | 10/min/IP   | Send a message, get AI response |
| GET    | `/health`      | None | 60/min/IP   | Health check + session count    |
| POST   | `/chat/clear`  | None | 60/min/IP   | Clear a session's history       |

## Run Locally

```bash
# From project root
python -m uvicorn agents.PreReleaseAgent.api.main:app --host 0.0.0.0 --port 8002
```

Or directly:

```bash
python agents/PreReleaseAgent/api/main.py
```

## Environment Variables

| Variable                              | Default       | Description                          |
|---------------------------------------|---------------|--------------------------------------|
| `CONDROX_PRERELEASE_PORT`             | `8002`        | Server port                          |
| `CONDROX_PRERELEASE_HOST`             | `0.0.0.0`     | Server host                          |
| `CONDROX_PRERELEASE_CORS_ORIGINS`     | localhost | Comma-separated CORS origins |
| `CONDROX_PRERELEASE_RATE_LIMIT`       | `10/minute`   | Rate limit for /chat                 |
| `CONDROX_PRERELEASE_RATE_LIMIT_DEFAULT` | `60/minute` | Rate limit for other endpoints       |

## Request / Response

### POST /chat

```json
// Request
{
  "message": "What is Condrox AI?",
  "session_id": null  // omit on first message, reuse returned ID for multi-turn
}

// Response
{
  "session_id": "pre_a1b2c3d4e5f6",
  "response": "Condrox AI is a cognitive conversational intelligence engine...",
  "intent": "explain",
  "domain": "ai",
  "mode": "explain",
  "confidence": 0.92,
  "truncated": false,
  "errors": []
}
```

## Isolation Guarantees

1. **Dedicated BrainController** — separate instance from main API
2. **`store_memory=False`** — zero persistent memory writes
3. **Ephemeral sessions** — in-memory only, TTL'd at 30 min
4. **No disk persistence** — nothing written to SQLite or files
5. **Per-session context** — users never see each other's history
