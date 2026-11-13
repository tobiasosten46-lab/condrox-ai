# Condrox API Reference

Base: `http://127.0.0.1:8000` (FastAPI + uvicorn).

## Authentication

All endpoints except `/health` require an API key.

```
X-API-Key: <CONDROX_API_KEY>
```

Or a session token from `POST /auth/login` (username + password, or the API
key as username). If `CONDROX_API_KEY` is unset, a random key is generated
per boot. Set `CONDROX_AUTH_ENABLED=0` to disable auth for local dev.

## Endpoints

### Core

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe. `{"status":"healthy","score":1.0}` |
| GET | `/status` | Runtime status — uptime, brain state, counters |
| POST | `/chat` | Conversational endpoint (see below) |
| POST | `/process` | Raw pipeline execution |
| POST | `/process/stream` | Streaming pipeline execution |
| POST | `/creative` | Emotion → music generation stack |
| GET | `/metrics` | Prometheus metrics |

### `POST /chat`

Request:

```json
{ "message": "what is a hash table?" }
```

Response:

```json
{
  "response": "...",
  "intent": "explain",
  "domain": "programming",
  "mode": "explain",
  "emotion": "neutral",
  "confidence": 0.81,
  "session_id": "chat-...",
  "response_time_ms": 45.2,
  "errors": []
}
```

### AI-OS introspection (`/aios/*`)

| Method | Path | Description |
|---|---|---|
| GET | `/aios/status` | Kernel + subsystem status |
| GET | `/aios/health` | Health of each OS service |
| GET | `/aios/events` | EventBus ring-buffer events |
| GET | `/aios/services` | Service lifecycle table |
| GET | `/aios/task-queue` | Queued/running tasks |
| GET | `/aios/data-flow` | Data-flow stats |
| POST | `/aios/shutdown` | Graceful shutdown |

### Monitoring

| Method | Path | Description |
|---|---|---|
| GET | `/monitoring/live` | Live interaction monitor |
| GET | `/monitoring/stats` | Aggregate interaction stats |

### Auth

| Method | Path | Description |
|---|---|---|
| POST | `/auth/login` | Create session token |
| POST | `/auth/logout` | Revoke session token |
| GET | `/auth/verify` | Validate token |

### Admin (`api/admin_routes.py`)

24 endpoints under `/admin/*` — health, config, memory CRUD, knowledge CRUD,
agent inspection, pipeline info, state save/load. All API-key gated. See the
router source for the exact list.

## Rate limits

Defaults (slowapi): `100/minute` global, `30/minute` on `/process`,
`10/minute` on `/creative`. Override via `CONDROX_RATE_LIMIT*` env vars.

## Errors

| Code | Meaning |
|---|---|
| 401 | Missing/invalid API key or token |
| 422 | Request-body validation failed |
| 429 | Rate limit exceeded |
