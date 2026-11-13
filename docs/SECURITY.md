# Condrox Security

## Threat model

Condrox is designed to be exposed to untrusted users. Layers, in order:

1. **Trusted hosts** — Host-header allowlist (PreReleaseAgent).
2. **CORS** — explicit origin allowlist; regex-restricted for the demo agent.
3. **Body size** — request-body cap (PreReleaseAgent).
4. **Rate limiting** — per-route limits via slowapi; per-session limits and
   automatic temporary IP bans in PreReleaseAgent.
5. **Authentication** — static `X-API-Key` (constant-time compare) or
   session tokens with expiry.
6. **Input security** — prompt-injection and malicious-pattern detection.
7. **Output security** — response scanned for PII (email, phone, card
   numbers), secret leakage, and injected-echo artifacts.
8. **Quality gate** — responses failing quality thresholds are held back.

## Hard rules for operators

- **Always set `CONDROX_API_KEY`** for anything reachable beyond localhost.
- Never run with `CONDROX_AUTH_ENABLED=0` on a network interface.
- Do not commit `.env`, SQLite databases, logs, or downloaded data — see
  `.gitignore`.
- The PreReleaseAgent's trusted-proxy mode (`CONDROX_PRERELEASE_CF_ONLY`)
  expects a fronting proxy that strips/spoofs `X-Forwarded-For` correctly;
  deploy it wrong and per-IP rate limiting is bypassable.
- `POST /aios/shutdown` exists — keep admin/API keys secret.

## Reporting

Open a private security advisory on the repository rather than a public
issue.
