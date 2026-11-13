# Condrox Configuration

All configuration is environment variables — every one has a safe default.
Copy `.env.example` to `.env` or export them in your shell.

## API server

| Variable | Default | Purpose |
|---|---|---|
| `CONDROX_API_KEY` | random per boot | Static API key for `X-API-Key` auth |
| `CONDROX_AUTH_ENABLED` | `1` | `0` disables auth (local dev only) |
| `CONDROX_API_HOST` | `0.0.0.0` | Bind address |
| `CONDROX_API_PORT` | `8000` | Bind port |
| `CONDROX_API_RELOAD` | — | uvicorn `--reload` toggle |
| `CONDROX_CORS_ORIGINS` | localhost ports | Comma-separated allowed origins |
| `CONDROX_RATE_LIMIT` | `100/minute` | Global rate limit |
| `CONDROX_RATE_LIMIT_PROCESS` | `30/minute` | `/process` limit |
| `CONDROX_RATE_LIMIT_CREATIVE` | `10/minute` | `/creative` limit |

## Storage

| Variable | Default | Purpose |
|---|---|---|
| `CONDROX_USER_DB` | `Databases/condrox_users.db` | User/session SQLite file |
| `CONDROX_OS_STATE_DB` | `Databases/condrox_os_state.db` | AI-OS state checkpoints |
| `CONDROX_KNOWLEDGE_DB` | `Databases/condrox_knowledge.db` | Knowledge SQLite file |
| `CONDROX_MONITORING_DIR` | — | Monitoring output dir |
| `CONDROX_REPORTS_DIR` | `Reports/evidence` | Evidence report dir (script) |

## Knowledge loading

| Variable | Default | Purpose |
|---|---|---|
| `CONDROX_KNOWLEDGE_ROOT` | `data/knowledge` | Chunked knowledge root |
| `CONDROX_SKIP_KNOWLEDGE_LOAD` | unset | `1` = boot with empty backend |
| `CONDROX_KL_MAX_FILES` | — | Cap files loaded per domain |
| `CONDROX_KL_MAX_CHUNKS` | `50` | Cap chunks per domain |
| `CONDROX_KL_MAX_CHUNK_CHARS` | `2000` | Cap chars per chunk |
| `CONDROX_SQLITE_KL_MAX_CHUNKS` | — | Cap chunks for SQLite loader |
| `KNOWLEDGE_DATA_PATH` | `data/knowledge` | Knowledge root used by the admin API |

## PreReleaseAgent (public demo)

| Variable | Default | Purpose |
|---|---|---|
| `CONDROX_PRERELEASE_HOST` / `PORT` | `127.0.0.1:8000` | Bind |
| `CONDROX_PRERELEASE_CORS_ORIGIN_REGEX` | localhost only | CORS regex |
| `CONDROX_PRERELEASE_RATE_LIMIT*` | `120/minute` | Demo rate limits |
| `CONDROX_PRERELEASE_MAX_BODY_SIZE` | — | Request body cap |
| `CONDROX_PRERELEASE_MAX_CONCURRENT` | — | Concurrency cap |
| `CONDROX_PRERELEASE_BAN_*` | — | Auto-ban thresholds |
| `CONDROX_PRERELEASE_TRUSTED_HOSTS` | — | Host header allowlist |
| `CONDROX_PRERELEASE_CF_ONLY` | — | Restrict to proxy IPs |
| `CONDROX_PRERELEASE_DISABLE_DOCS` | — | Hide OpenAPI docs |
| `CONDROX_PRERELEASE_DISABLE_LOCK` | `1` | Single-instance lock off |

## DeveloperAgent

| Variable | Default | Purpose |
|---|---|---|
| `CONDROX_RULESET_PATH` | `ruleset` | Ruleset ingestion source dir |
| `CONDROX_ALLOW_EXTERNAL` | — | Permit external resources |

## Desktop app (`app/`)

| Variable | Default | Purpose |
|---|---|---|
| `CONDROX_PORT` | `8765` | Backend port the app spawns |
| `CONDROX_PYTHON` | auto-detect | Python interpreter path |
