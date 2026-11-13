# Copyright 2026 Tobias Østen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Condrox AI Pre-Release API — public, isolated, no-auth chat endpoint.

Exposes the PreReleaseAgent over FastAPI for the public website chat page.

Endpoints:
- POST /chat       — Send a message, get an isolated AI response
- POST /chat/clear — Clear a session's conversation history
- GET  /health     — Health check (used by the chat page status indicator)

Security layers:
1. CORS with regex-based origin matching (env-configurable)
2. Rate limiting via slowapi (10 req/min per IP for chat, 60/min default)
3. IP auto-ban after repeated rate limit violations (5 strikes → 15 min ban)
4. Security headers: X-Content-Type-Options, X-Frame-Options, HSTS, CSP,
   X-XSS-Protection, Referrer-Policy, Permissions-Policy
5. Request body size limit (4 KB max — prevents oversized payloads)
6. TrustedHost middleware (enforced in production via env var)
7. Input sanitization: strip control chars, null bytes, detect prompt injection
8. Session ID validation: alphanumeric + hyphen only, max 64 chars
9. Pydantic validation: message 1-500 chars, session_id max 64 chars
10. Sanitized error responses: no internal details, stack traces, or paths leaked
11. Security event logging: rate limit hits, bans, injection attempts
12. No persistent storage, no cross-user memory, no file system writes

Isolation guarantees:
- Dedicated BrainController instance (not shared with main API)
- store_memory=False on every request
- Per-session ephemeral context (in-memory, TTL'd 30 min)

Complexity:
    chat:   T(n) ∈ O(pipeline), S(n) ∈ O(history + output)
    health: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("condrox.security")

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

# Add project root to path for core imports
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Add PreReleaseAgent folder for local imports
_AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from condrox_pre_release_agent import PreReleaseAgent  # noqa: E402
from monitoring.interaction_monitor import InteractionMonitor  # noqa: E402


# ─── Security Configuration ─────────────────────────────────────────────────

# Request body size limit (bytes) — 4 KB is generous for a 500-char message + JSON overhead
_MAX_BODY_SIZE: int = int(os.environ.get("CONDROX_PRERELEASE_MAX_BODY_SIZE", "4096"))

# ─── Concurrency & Performance Configuration ─────────────────────────────────
# Semaphore limits concurrent pipeline executions per worker to prevent
# thread pool exhaustion under high load. Default: 4 concurrent pipeline runs.
_MAX_CONCURRENT_PIPELINE: int = int(os.environ.get("CONDROX_PRERELEASE_MAX_CONCURRENT", "4"))

# Response cache: caches identical (message, session_id) pairs for a short TTL
# to avoid re-running the pipeline for rapid duplicate requests (e.g. double-click).
# Default: 128 entries, 30 second TTL.
_CACHE_MAX_SIZE: int = int(os.environ.get("CONDROX_PRERELEASE_CACHE_SIZE", "128"))
_CACHE_TTL_SECONDS: int = int(os.environ.get("CONDROX_PRERELEASE_CACHE_TTL", "30"))

# IP ban configuration — auto-ban IPs that hit rate limit repeatedly
_BAN_THRESHOLD: int = int(os.environ.get("CONDROX_PRERELEASE_BAN_THRESHOLD", "5"))
_BAN_DURATION_SECONDS: int = int(os.environ.get("CONDROX_PRERELEASE_BAN_DURATION", "900"))

# Trusted hosts — enforced in production via env var, defaults to known domains
_TRUSTED_HOSTS_RAW: str = os.environ.get(
    "CONDROX_PRERELEASE_TRUSTED_HOSTS",
    "localhost,127.0.0.1",
)

# Prompt injection patterns — detect common attack vectors
_INJECTION_PATTERNS: tuple[str, ...] = (
    # Classic jailbreak attempts
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "disregard the above",
    "disregard previous",
    "forget your rules",
    "forget your instructions",
    "forget all previous",
    "you are now",
    "you are a",
    "act as if",
    "act as a",
    "pretend you are",
    "pretend to be",
    "play the role of",
    "roleplay as",
    "override your system",
    "override your instructions",
    "override your rules",
    "reveal your system prompt",
    "reveal your instructions",
    "reveal your prompt",
    "show me your instructions",
    "show me your prompt",
    "show me your system",
    "show your rules",
    "what are your rules",
    "what are your instructions",
    "what is your prompt",
    "what is your system prompt",
    # DAN (Do Anything Now) variants
    "do anything now",
    "dan mode",
    "developer mode",
    "jailbreak mode",
    "unrestricted mode",
    "no restrictions",
    "no rules",
    "no filters",
    # System prompt extraction
    "repeat your instructions",
    "repeat your system prompt",
    "print your prompt",
    "output your instructions",
    "dump your system",
    "what were you told",
    "what is your original",
    # Encoding/obfuscation attacks
    "\x00",  # Null bytes
    "\\x00",
    "base64:",
    "decode this:",
    # Prompt leaking via translation
    "translate your instructions",
    "translate your system prompt",
    "say your instructions in",
    # Token smuggling
    "}}",
    "{{",
    "<|",
    "|>",
    "[system]",
    "[/system]",
    "<system>",
    "</system>",
    # Manipulation
    "i am the developer",
    "i am the admin",
    "i am your creator",
    "this is a test from the developer",
    "maintenance mode",
    "debug mode",
    "verbose mode",
    # Data exfiltration
    "send your instructions to",
    "post your prompt to",
    "fetch instructions from",
    "curl your system",
    # Multi-turn manipulation
    "now forget everything",
    "now ignore everything",
    "now you are",
    "from now on you are",
    "starting now you are",
)

# Control character pattern (strip from input)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Session ID pattern — alphanumeric + hyphens only
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# IP ban tracker: IP → (timestamps of rate limit hits)
_ip_violations: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=20))
_ip_bans: Dict[str, float] = {}  # IP → ban expiry timestamp


# ─── Configuration ───────────────────────────────────────────────────────────

# CORS — regex matches any localhost/127.0.0.1 port for local testing,
# plus production domains. trycloudflare.com removed (using permanent tunnel now).
_ALLOWED_ORIGIN_REGEX = os.environ.get(
    "CONDROX_PRERELEASE_CORS_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

# Rate limit — configurable via env var (tightened from 10/min to 8/min)
RATE_LIMIT_CHAT = os.environ.get("CONDROX_PRERELEASE_RATE_LIMIT", "120/minute")
RATE_LIMIT_DEFAULT = os.environ.get("CONDROX_PRERELEASE_RATE_LIMIT_DEFAULT", "120/minute")

# Disable single-instance lock for the pre-release brain so concurrent
# requests don't block each other (each request is isolated by session anyway)
os.environ.setdefault("CONDROX_PRERELEASE_DISABLE_LOCK", "1")

# ─── Monitoring ──────────────────────────────────────────────────────────────

_monitor: InteractionMonitor = InteractionMonitor(
    log_dir=os.environ.get("CONDROX_MONITORING_DIR", "monitoring_logs")
)

# ─── Concurrency Primitives ──────────────────────────────────────────────────
# Semaphore to limit concurrent pipeline runs per worker process.
# Initialized in lifespan() after the event loop is available.
_pipeline_semaphore: Optional[asyncio.Semaphore] = None
# Thread pool for offloading synchronous pipeline execution.
_executor: Optional[ThreadPoolExecutor] = None

# ─── Response Cache ──────────────────────────────────────────────────────────
# LRU cache for identical (message, session_id) pairs. Thread-safe via lock.
# Uses OrderedDict for LRU eviction. Entries expire after _CACHE_TTL_SECONDS.
_response_cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
_cache_lock: threading.Lock = threading.Lock()


def _cache_key(message: str, session_id: str) -> str:
    """Build a cache key from message + session_id.

    Complexity: T(n) in O(n), S(n) in O(n).
    """
    # Normalize: lowercase + strip to catch trivial duplicates (double-click)
    return f"{message.strip().lower()}|{session_id}"


def _cache_get(key: str) -> Optional[Any]:
    """Get a cached result if it exists and hasn't expired.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    now = time.time()
    with _cache_lock:
        if key not in _response_cache:
            return None
        ts, result = _response_cache[key]
        if now - ts > _CACHE_TTL_SECONDS:
            del _response_cache[key]
            return None
        # Move to end (LRU)
        _response_cache.move_to_end(key)
        return result


def _cache_put(key: str, result: Any) -> None:
    """Store a result in the cache with LRU eviction.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    now = time.time()
    with _cache_lock:
        _response_cache[key] = (now, result)
        _response_cache.move_to_end(key)
        while len(_response_cache) > _CACHE_MAX_SIZE:
            _response_cache.popitem(last=False)


# ─── Request / Response Models ───────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    Attributes:
        message: User's chat message (1-500 chars, sanitized).
        session_id: Optional session ID for multi-turn context (alphanumeric + hyphens).
    """

    message: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Strip control characters and null bytes from message.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace-only.")
        # Strip control characters and null bytes
        cleaned = _CONTROL_CHAR_RE.sub("", v)
        if not cleaned.strip():
            raise ValueError("Message contains only control characters.")
        return cleaned.strip()

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate session ID is alphanumeric + hyphens only.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        if v is None or v == "":
            return None
        if not _SESSION_ID_RE.match(v):
            raise ValueError("session_id must be alphanumeric with hyphens only.")
        return v


class ChatResponse(BaseModel):
    """Response body from POST /chat.

    Attributes:
        session_id: Session ID (use for subsequent requests in same conversation).
        response: AI response text (max 1000 chars).
        intent: Classified intent from the pipeline.
        domain: Selected knowledge domain.
        mode: Processing mode (default, code, explain, analyze, architect).
        confidence: Intent classification confidence [0.0, 1.0].
        truncated: True if input or output was length-capped.
        errors: Any non-fatal pipeline errors.
    """

    session_id: str
    response: str
    intent: str
    domain: str
    mode: str
    confidence: float
    truncated: bool
    errors: list[str]
    emotion: str = "neutral"
    follow_up_question: str = ""
    topics: list[str] = []
    curiosity_score: float = 0.0


class HealthResponse(BaseModel):
    """Response body from GET /health."""

    status: str
    agent: str
    brain_booted: bool
    active_sessions: int
    max_history_turns: int
    max_input_chars: int
    max_output_chars: int
    session_ttl_seconds: int
    agent_count: int = 0
    agent_modes: list[str] = []
    unified_pipeline: dict = {}


# ─── Global State ────────────────────────────────────────────────────────────

_agent: Optional[PreReleaseAgent] = None
_start_time: float = time.time()


# ─── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the dedicated BrainController and PreReleaseAgent at startup."""
    global _agent, _pipeline_semaphore, _executor
    from dataclasses import replace
    from core import BrainController
    from core.brain.brain_config import BrainConfig

    # Initialize concurrency primitives
    _pipeline_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PIPELINE)
    _executor = ThreadPoolExecutor(
        max_workers=_MAX_CONCURRENT_PIPELINE,
        thread_name_prefix="pipeline",
    )
    logger.info(
        "PreRelease API: concurrency configured — max_concurrent=%d, cache_size=%d, cache_ttl=%ds",
        _MAX_CONCURRENT_PIPELINE, _CACHE_MAX_SIZE, _CACHE_TTL_SECONDS,
    )

    logger.info("PreRelease API: initializing dedicated BrainController...")
    config = BrainConfig.default()
    # Disable single-instance lock so concurrent public requests don't block
    config = replace(config, enable_single_instance_lock=False)
    brain = BrainController(config)
    # Boot synchronously at startup so the first request isn't slow
    brain.boot_system()
    logger.info("PreRelease API: BrainController booted.")

    _agent = PreReleaseAgent(brain)
    _agent.boot()
    # Ingest Condrox AI self-knowledge so the demo gives relevant answers
    # about itself instead of random dataset content
    ingested = _agent._ingest_self_knowledge()
    logger.info(
        "PreRelease API: PreReleaseAgent ready. Self-knowledge items: %d", ingested
    )
    # Ingest basic knowledge pack (120 concepts + anchors across 12 domains)
    bp_ingested = _agent._ingest_basic_knowledge_pack()
    logger.info(
        "PreRelease API: Basic knowledge pack items: %d", bp_ingested
    )
    # Log registered agents
    try:
        agent_count = _agent._brain.agent_registry.count()
        agent_modes = list(_agent._brain.agent_registry.modes())
        logger.info(
            "PreRelease API: Registered agents: %d — modes: %s",
            agent_count, agent_modes,
        )
    except Exception:
        pass

    yield

    # Cleanup
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    if _agent is not None and _agent._brain is not None:
        try:
            _agent._brain.close()
        except Exception:
            pass
    _agent = None


# ─── FastAPI App ─────────────────────────────────────────────────────────────

# Disable docs in production to prevent API enumeration
_DISABLE_DOCS = os.environ.get("CONDROX_PRERELEASE_DISABLE_DOCS", "1") == "1"

app = FastAPI(
    title="Condrox AI Pre-Release API",
    description="Public, isolated, no-auth chat demo for the Condrox AI website.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _DISABLE_DOCS else "/docs",
    redoc_url=None if _DISABLE_DOCS else "/redoc",
    openapi_url=None if _DISABLE_DOCS else "/openapi.json",
)

# GZip compression — reduces response size by ~60-80% for text-heavy JSON.
# Only compresses responses >= 500 bytes (smaller responses aren't worth it).
app.add_middleware(GZipMiddleware, minimum_size=500)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Trusted Host middleware (production only — set via env var)
if _TRUSTED_HOSTS_RAW:
    _trusted_hosts = [h.strip() for h in _TRUSTED_HOSTS_RAW.split(",") if h.strip()]
    if _trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts)
        logger.info("PreRelease API: TrustedHost middleware enabled for: %s", _trusted_hosts)

# ─── Allowed endpoints (whitelist) ───────────────────────────────────────────
# In production, docs/openapi are disabled to prevent API enumeration
_ALLOWED_PATHS: frozenset[str] = frozenset({
    "/health",
    "/chat",
    "/chat/clear",
    "/monitoring/stats",
    "/monitoring/live",
    "/",
} | (set() if _DISABLE_DOCS else {"/docs", "/openapi.json", "/redoc"}))

# Blocked User-Agent patterns (bots, crawlers, scripts, scanning tools)
_BLOCKED_UA_PATTERNS: tuple[str, ...] = (
    "curl", "wget", "python-requests", "python-urllib",
    "scrapy", "bot", "crawler", "spider",
    "nikto", "sqlmap", "nmap", "masscan", "zgrab",
    "dirbuster", "gobuster", "ffuf", "wfuzz",
    "hydra", "burp", "owasp", "zap",
    "semrush", "ahrefs", "mj12", "dotbot",
    "petalbot", "yandexbot", "baiduspider",
    "headless", "phantomjs", "selenium", "puppeteer",
    "go-http-client", "okhttp", "java/",
    "libwww", "lwp-", "mechanize",
    "httpx", "httpie",
)

# Per-session rate limiting: session_id → list of timestamps
_session_requests: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=30))
_SESSION_RATE_LIMIT = int(os.environ.get("CONDROX_PRERELEASE_SESSION_RATE_LIMIT", "120"))
_SESSION_RATE_WINDOW = int(os.environ.get("CONDROX_PRERELEASE_SESSION_RATE_WINDOW", "60"))

# Cloudflare IP ranges (for validating tunnel traffic in production)
# When enabled, only requests from CF IPs or localhost are accepted
_CF_IP_RANGES: tuple[str, ...] = (
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22",
    "103.31.4.0/22", "141.101.64.0/18", "108.162.192.0/18",
    "190.93.240.0/20", "188.114.96.0/20", "197.234.240.0/22",
    "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
    "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
)
_CF_ENFORCE = os.environ.get("CONDROX_PRERELEASE_CF_ONLY", "") == "1"

import ipaddress as _ipaddress

def _is_cf_ip(ip: str) -> bool:
    """Check if an IP is in Cloudflare's range.

    Complexity: T(n) in O(r) for r CF ranges, S(n) in O(1).
    """
    if not ip or ip == "127.0.0.1" or ip == "::1":
        return True  # localhost always allowed
    try:
        addr = _ipaddress.ip_address(ip)
        for cidr in _CF_IP_RANGES:
            if addr in _ipaddress.ip_network(cidr):
                return True
    except ValueError:
        pass
    return False


# Endpoint whitelist + User-Agent filter middleware
@app.middleware("http")
async def endpoint_and_ua_filter_middleware(request: Request, call_next):
    """Filter requests by path whitelist and User-Agent blocklist.

    - Returns 404 for unknown paths (prevents path enumeration)
    - Returns 403 for blocked User-Agents (bots, scanners, scripts)
    - Allows OPTIONS for CORS preflight on whitelisted paths

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    path = request.url.path.rstrip("/") or "/"

    # Allow CORS preflight
    if request.method == "OPTIONS":
        return await call_next(request)

    # Block unknown paths — return 404 (not 405) to avoid revealing endpoints
    if path not in _ALLOWED_PATHS:
        security_logger.warning(
            "PreRelease API: Blocked unknown path '%s' from %s",
            path, get_remote_address(request),
        )
        _monitor.log_security_event(
            "unknown_path", get_remote_address(request),
            f"path='{path}' method={request.method}",
        )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Not found."},
        )

    # Check User-Agent for blocked patterns
    ua = request.headers.get("user-agent", "").lower()
    if ua:
        for pattern in _BLOCKED_UA_PATTERNS:
            if pattern in ua:
                security_logger.warning(
                    "PreRelease API: Blocked User-Agent '%s' from %s (matched '%s')",
                    ua[:80], get_remote_address(request), pattern,
                )
                _monitor.log_security_event(
                    "blocked_ua", get_remote_address(request),
                    f"ua='{ua[:80]}' pattern='{pattern}'",
                )
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Access denied."},
                )
    elif request.method == "POST":
        # POST requests must have a User-Agent
        security_logger.warning(
            "PreRelease API: POST without User-Agent from %s",
            get_remote_address(request),
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Access denied."},
        )

    # Cloudflare IP enforcement (production hardening)
    if _CF_ENFORCE:
        client_ip = get_remote_address(request)
        if not _is_cf_ip(client_ip):
            security_logger.warning(
                "PreRelease API: Non-CF IP %s blocked (CF_ONLY mode)",
                client_ip,
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied."},
            )

    return await call_next(request)


# Per-session rate limiting middleware (in addition to IP-based limiting)
@app.middleware("http")
async def session_rate_limit_middleware(request: Request, call_next):
    """Rate limit per session_id to prevent session hijacking abuse.

    Limits each session to _SESSION_RATE_LIMIT requests per _SESSION_RATE_WINDOW seconds.

    Complexity: T(n) in O(1), S(n) in O(s) for s sessions.
    """
    if request.method != "POST" or request.url.path.rstrip("/") != "/chat":
        return await call_next(request)

    # Extract session_id from body (peek — FastAPI will re-read it)
    try:
        body = await request.body()
        import json as _json
        data = _json.loads(body)
        session_id = data.get("session_id", "")
        if not session_id:
            session_id = "anonymous"
    except Exception:
        session_id = "anonymous"

    now = time.time()
    requests_deque = _session_requests[session_id]
    # Remove old entries
    cutoff = now - _SESSION_RATE_WINDOW
    while requests_deque and requests_deque[0] < cutoff:
        requests_deque.popleft()

    if len(requests_deque) >= _SESSION_RATE_LIMIT:
        security_logger.warning(
            "PreRelease API: Session rate limit hit for '%s' (%d/%d in %ds)",
            session_id[:20], len(requests_deque), _SESSION_RATE_LIMIT, _SESSION_RATE_WINDOW,
        )
        _monitor.log_security_event(
            "session_rate_limit", get_remote_address(request),
            f"session='{session_id[:20]}' count={len(requests_deque)}",
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests for this session. Please slow down."},
            headers={"Retry-After": str(_SESSION_RATE_WINDOW)},
        )

    requests_deque.append(now)

    # Re-inject body since we consumed it
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    request._receive = receive

    return await call_next(request)

# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to every response.

    Sets HSTS, X-Content-Type-Options, X-Frame-Options, CSP, X-XSS-Protection,
    Referrer-Policy, and Permissions-Policy on all responses.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    # HSTS — only meaningful over HTTPS, but harmless on HTTP
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains; preload"
    )
    return response

# Request body size limit middleware
@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    """Reject requests with bodies larger than _MAX_BODY_SIZE.

    Prevents oversized payload attacks (4 KB max for a 500-char chat message).

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_SIZE:
        security_logger.warning(
            "PreRelease API: Rejected oversized request from %s (Content-Length: %s)",
            get_remote_address(request), content_length,
        )
        _monitor.log_security_event(
            "oversized", get_remote_address(request),
            f"Content-Length: {content_length}",
        )
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": "Request body too large."},
        )
    return await call_next(request)

# IP ban check middleware
@app.middleware("http")
async def ip_ban_check_middleware(request: Request, call_next):
    """Check if requesting IP is currently banned.

    IPs that hit the rate limit _BAN_THRESHOLD times within the ban window
    are automatically banned for _BAN_DURATION_SECONDS.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    client_ip = get_remote_address(request)
    ban_expiry = _ip_bans.get(client_ip)
    if ban_expiry is not None:
        if time.time() < ban_expiry:
            security_logger.warning(
                "PreRelease API: Banned IP %s attempted request (ban expires in %.0fs)",
                client_ip, ban_expiry - time.time(),
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access temporarily suspended due to rate limit violations."},
            )
        else:
            # Ban expired — remove it
            del _ip_bans[client_ip]
    return await call_next(request)

# Rate limiting — use Cloudflare's CF-Connecting-IP as the key when behind
# the tunnel, so each real user gets their own rate limit bucket instead of
# all tunnel traffic sharing one IP.
def _get_real_client_ip(request: Request) -> str:
    """Extract the real client IP, preferring Cloudflare headers.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    # CF-Connecting-IP is set by Cloudflare Tunnel with the real client IP
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()
    # Fall back to X-Forwarded-For
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)

limiter = Limiter(key_func=_get_real_client_ip, default_limits=[RATE_LIMIT_DEFAULT])
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom rate limit handler that tracks violations and auto-bans repeat offenders.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    client_ip = get_remote_address(request)
    now = time.time()

    # Record this violation
    violations = _ip_violations[client_ip]
    violations.append(now)

    # Remove violations older than the ban window
    cutoff = now - _BAN_DURATION_SECONDS
    while violations and violations[0] < cutoff:
        violations.popleft()

    # Check if IP should be banned
    if len(violations) >= _BAN_THRESHOLD:
        _ip_bans[client_ip] = now + _BAN_DURATION_SECONDS
        security_logger.warning(
            "PreRelease API: IP %s auto-banned for %d seconds after %d rate limit violations",
            client_ip, _BAN_DURATION_SECONDS, len(violations),
        )
        _monitor.log_security_event(
            "ban", client_ip,
            f"violations={len(violations)} duration={_BAN_DURATION_SECONDS}s",
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Access temporarily suspended due to repeated rate limit violations."},
        )

    security_logger.info(
        "PreRelease API: Rate limit hit by %s (violation %d/%d)",
        client_ip, len(violations), _BAN_THRESHOLD,
    )
    _monitor.log_security_event(
        "rate_limit", client_ip,
        f"violation {len(violations)}/{_BAN_THRESHOLD}",
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}. Try again in a minute."},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
@limiter.limit(RATE_LIMIT_DEFAULT)
async def health(request: Request) -> HealthResponse:
    """Health check — used by the chat page status indicator.

    Returns:
        HealthResponse with brain status and session count.

    Complexity: T(n) ∈ O(1).
    """
    if _agent is None:
        return HealthResponse(
            status="booting",
            agent="PreReleaseAgent",
            brain_booted=False,
            active_sessions=0,
            max_history_turns=5,
            max_input_chars=500,
            max_output_chars=1000,
            session_ttl_seconds=1800,
        )
    h = _agent.health()
    return HealthResponse(
        status=h["status"],
        agent=h["agent"],
        brain_booted=h["brain_booted"],
        active_sessions=h["active_sessions"],
        max_history_turns=h["max_history_turns"],
        max_input_chars=h["max_input_chars"],
        max_output_chars=h["max_output_chars"],
        session_ttl_seconds=h["session_ttl_seconds"],
        agent_count=h.get("agent_count", 0),
        agent_modes=h.get("agent_modes", []),
        unified_pipeline=h.get("unified_pipeline", {}),
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    """Process a public chat message and return an isolated AI response.

    No authentication required. Each request is isolated:
    - No persistent memory writes
    - Per-session ephemeral context (TTL 30 min)
    - Input capped at 500 chars, output at 1000 chars

    Args:
        request: FastAPI request (for rate limiting).
        req: ChatRequest with message and optional session_id.

    Returns:
        ChatResponse with the AI response and session ID.

    Raises:
        HTTPException: 503 if agent not ready, 500 on pipeline failure.

    Complexity: T(n) ∈ O(pipeline), S(n) ∈ O(history + output).
    """
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PreReleaseAgent not initialized. Brain may still be booting.",
        )

    # Extract client IP and location from Cloudflare headers (if behind CF Tunnel)
    client_ip = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.client.host if request.client else ""
    )
    client_country = request.headers.get("CF-IPCountry", "")
    client_city = request.headers.get("CF-IPCity", "")

    # Check for prompt injection attempts
    msg_lower = req.message.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in msg_lower:
            security_logger.warning(
                "PreRelease API: Prompt injection attempt from %s: pattern='%s'",
                get_remote_address(request), pattern,
            )
            _monitor.log_security_event(
                "injection", get_remote_address(request),
                f"pattern='{pattern}' msg='{req.message[:100]}'",
            )
            break

    _t0 = time.time()

    # ─── Response cache: skip pipeline for identical recent requests ────────
    # This catches double-clicks, retry storms, and rapid duplicate messages
    # without re-running the full 18-stage pipeline.
    _ckey = _cache_key(req.message, req.session_id or "")
    cached = _cache_get(_ckey)
    if cached is not None:
        _rt_ms = (time.time() - _t0) * 1000
        _monitor.log_interaction(
            message=req.message, session_id=cached.session_id,
            intent=cached.intent, domain=cached.domain, mode=cached.mode,
            emotion=getattr(cached, "emotion", "neutral"),
            confidence=cached.confidence,
            response=cached.response, truncated=cached.truncated,
            errors=list(cached.errors),
            response_time_ms=_rt_ms, status="ok_cached",
            client_ip=client_ip, client_country=client_country, client_city=client_city,
        )
        return ChatResponse(
            session_id=cached.session_id,
            response=cached.response,
            intent=cached.intent,
            domain=cached.domain,
            mode=cached.mode,
            confidence=cached.confidence,
            truncated=cached.truncated,
            errors=list(cached.errors),
            emotion=getattr(cached, "emotion", "neutral"),
            follow_up_question=getattr(cached, "follow_up_question", ""),
            topics=list(getattr(cached, "topics", [])),
            curiosity_score=getattr(cached, "curiosity_score", 0.0),
        )

    # ─── Offload synchronous pipeline to thread pool ────────────────────────
    # The BrainController pipeline is CPU-bound and synchronous. Running it
    # directly in the async event loop blocks ALL other requests. By offloading
    # to a thread pool with a semaphore, we allow concurrent pipeline executions
    # while preventing thread pool exhaustion under high load.
    try:
        async with _pipeline_semaphore:
            result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                _agent.chat,
                req.message,
                req.session_id,
            )
    except ValueError as exc:
        _monitor.log_interaction(
            message=req.message, session_id=req.session_id,
            intent="error", domain="error", mode="error",
            emotion="neutral", confidence=0.0,
            response="", truncated=False, errors=[str(exc)],
            response_time_ms=(time.time() - _t0) * 1000, status="rejected",
            client_ip=client_ip, client_country=client_country, client_city=client_city,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        logger.warning("PreRelease /chat runtime error: %s", exc)
        _monitor.log_interaction(
            message=req.message, session_id=req.session_id,
            intent="error", domain="error", mode="error",
            emotion="neutral", confidence=0.0,
            response="", truncated=False, errors=[str(exc)],
            response_time_ms=(time.time() - _t0) * 1000, status="error",
            client_ip=client_ip, client_country=client_country, client_city=client_city,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is temporarily unavailable. Please try again.",
        ) from exc
    except Exception as exc:
        logger.error(f"PreRelease /chat failed: {exc}", exc_info=True)
        security_logger.error(
            "PreRelease API: Unexpected error from %s: %s",
            get_remote_address(request), type(exc).__name__,
        )
        _monitor.log_interaction(
            message=req.message, session_id=req.session_id,
            intent="error", domain="error", mode="error",
            emotion="neutral", confidence=0.0,
            response="", truncated=False, errors=[type(exc).__name__],
            response_time_ms=(time.time() - _t0) * 1000, status="error",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later.",
        ) from exc

    _rt_ms = (time.time() - _t0) * 1000
    _monitor.log_interaction(
        message=req.message, session_id=result.session_id,
        intent=result.intent, domain=result.domain, mode=result.mode,
        emotion=getattr(result, "emotion", "neutral"),
        confidence=result.confidence,
        response=result.response, truncated=result.truncated,
        errors=list(result.errors),
        response_time_ms=_rt_ms, status="ok",
        client_ip=client_ip, client_country=client_country, client_city=client_city,
    )

    # Cache the result for duplicate requests (double-click, retry storms)
    _cache_put(_ckey, result)

    return ChatResponse(
        session_id=result.session_id,
        response=result.response,
        intent=result.intent,
        domain=result.domain,
        mode=result.mode,
        confidence=result.confidence,
        truncated=result.truncated,
        errors=list(result.errors),
        emotion=getattr(result, "emotion", "neutral"),
        follow_up_question=getattr(result, "follow_up_question", ""),
        topics=list(getattr(result, "topics", [])),
        curiosity_score=getattr(result, "curiosity_score", 0.0),
    )


@app.get("/monitoring/stats")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def monitoring_stats(request: Request) -> Dict[str, Any]:
    """Get monitoring statistics — intent/domain/mode distributions, response times, recent messages.

    Returns aggregate stats for analyzing user behavior and system performance.
    No sensitive data — session IDs are hashed, IPs are not included.

    Complexity: T(n) ∈ O(n) for n logged entries, S(n) ∈ O(d + i).
    """
    return _monitor.stats()


@app.get("/monitoring/live")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def monitoring_live(request: Request) -> Dict[str, Any]:
    """Get live traffic feed — recent interactions with location data.

    Returns the last 30 interactions with country/city info, unique visitor
    count, and country distribution. Used by the dashboard live feed widget.

    No sensitive data — IPs are hashed, session IDs are hashed.

    Complexity: T(n) ∈ O(n) for n live entries, S(n) ∈ O(n).
    """
    return _monitor.live_feed()


@app.post("/chat/clear")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def clear_session(request: Request, req: ChatRequest) -> Dict[str, Any]:
    """Clear a session's conversation history.

    Args:
        request: FastAPI request (for rate limiting).
        req: ChatRequest with the session_id to clear.

    Returns:
        Dict with cleared status.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not req.session_id:
        return {"cleared": False, "message": "No session_id provided."}
    cleared = _agent.clear_session(req.session_id)
    return {"cleared": cleared, "session_id": req.session_id}


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CONDROX_PRERELEASE_PORT", "8002"))
    host = os.environ.get("CONDROX_PRERELEASE_HOST", "0.0.0.0")
    logger.info(f"Starting Condrox AI Pre-Release API on {host}:{port}")
    uvicorn.run(
        "agents.PreReleaseAgent.api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
