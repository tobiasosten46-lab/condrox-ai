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

"""
FastAPI REST API for the unified Brain system.

Exposes /health, /process, /metrics, /status, and /creative endpoints.
Implements API key authentication, rate limiting, and CORS.

Security:
- API key authentication via X-API-Key header (configurable via env var)
- Rate limiting via slowapi (configurable limits per endpoint)
- CORS configuration (configurable allowed origins)
- Input validation via Pydantic models

Mathematical Foundation:
- Endpoint latency is dominated by the 14-stage pipeline: O(Î£ s_i).
- Rate limiting uses token bucket: O(1) per request.
- API key validation: O(1) constant-time comparison.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core import BrainController
from core import __version__ as CORE_VERSION
from core.storage.user_database import UserDatabase
from api.admin_routes import router as admin_router
from api.monitoring import get_tracker

# â”€â”€â”€ Security Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# API key â€” read from environment variable, generate a random one if not set
# In production, ALWAYS set CONDROX_API_KEY environment variable
_DEFAULT_API_KEY = secrets.token_urlsafe(32)
API_KEY: str = os.environ.get("CONDROX_API_KEY", _DEFAULT_API_KEY)

# CORS allowed origins â€” comma-separated in env var, defaults to localhost
_ALLOWED_ORIGINS_RAW = os.environ.get(
    "CONDROX_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS: list[str] = [
    o.strip() for o in _ALLOWED_ORIGINS_RAW.split(",") if o.strip()
]

# Rate limit settings â€” configurable via env var
RATE_LIMIT_DEFAULT = os.environ.get("CONDROX_RATE_LIMIT", "100/minute")
RATE_LIMIT_PROCESS = os.environ.get("CONDROX_RATE_LIMIT_PROCESS", "30/minute")
RATE_LIMIT_CREATIVE = os.environ.get("CONDROX_RATE_LIMIT_CREATIVE", "10/minute")

# Whether auth is enabled (set CONDROX_AUTH_ENABLED=0 to disable for dev)
AUTH_ENABLED: bool = os.environ.get("CONDROX_AUTH_ENABLED", "1") != "0"

# â”€â”€â”€ API Key Authentication â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> str:
    """Verify the API key or session token from the X-API-Key header.

    Accepts either:
    - Static API key (constant-time comparison)
    - Session token from /auth/login (validated against user_db)

    Args:
        api_key: API key or session token from request header.

    Returns:
        The verified key/token.

    Raises:
        HTTPException: 401 if authentication is enabled and key is missing or invalid.

    Complexity: T(n) âˆˆ O(1) â€” constant-time comparison + dict lookup.
    """
    if not AUTH_ENABLED:
        return "auth-disabled"

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check static API key first
    if hmac.compare_digest(api_key, API_KEY):
        return api_key

    # Check session token from user database
    if user_db.validate_token(api_key) is not None:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key or session token.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


# â”€â”€â”€ Rate Limiting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_DEFAULT])


# â”€â”€â”€ Prometheus metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "condrox_requests_total",
    "Total processed requests",
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "condrox_request_latency_seconds",
    "Request processing latency in seconds",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    registry=REGISTRY,
)
SAFETY_BLOCKS = Counter(
    "condrox_safety_blocks_total",
    "Total requests blocked by safety layer",
    registry=REGISTRY,
)
HEALTH_SCORE = Gauge(
    "condrox_health_score",
    "Current health score [0.0, 1.0]",
    registry=REGISTRY,
)
MEMORY_COUNT = Gauge(
    "condrox_memory_items",
    "Number of stored memory items",
    registry=REGISTRY,
)
PIPELINE_STAGES = Gauge(
    "condrox_pipeline_stages",
    "Number of pipeline stages",
    registry=REGISTRY,
)
ACTIVE_GOALS = Gauge(
    "condrox_active_goals",
    "Number of active dialogue goals",
    registry=REGISTRY,
)
TOKEN_USAGE = Gauge(
    "condrox_token_usage",
    "Current token usage",
    registry=REGISTRY,
)
PRIORITY_QUEUE_SIZE = Gauge(
    "condrox_priority_queue_size",
    "Current priority queue size",
    registry=REGISTRY,
)
AUTH_FAILURES = Counter(
    "condrox_auth_failures_total",
    "Total authentication failures",
    registry=REGISTRY,
)
RATE_LIMIT_HITS = Counter(
    "condrox_rate_limit_hits_total",
    "Total requests rejected by rate limiter",
    registry=REGISTRY,
)


# â”€â”€â”€ Request / Response models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class ProcessRequest(BaseModel):
    """Request body for /process.

    Attributes:
        input: User text to process.
        context: Optional key-value context.
    """

    input: str = Field(..., min_length=1, max_length=4096)
    context: Dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    """Response body from /process.

    Attributes:
        request_id: Unique request identifier.
        final_output: Rendered response text.
        mode: Processing mode.
        domain: Selected domain.
        intent: Classified intent.
        intent_confidence: Confidence of intent classification.
        security_approved: Security check result.
        ethical_approved: Ethics check result.
        errors: Any non-fatal errors.
    """

    request_id: str
    final_output: str
    mode: str
    domain: str
    intent: str
    intent_confidence: float
    security_approved: bool
    ethical_approved: bool
    errors: list[str]


class CreativeRequest(BaseModel):
    """Request body for /creative.

    Attributes:
        input: User text to process.
        include_lyrics: Whether to generate lyrics.
    """

    input: str = Field(..., min_length=1, max_length=4096)
    include_lyrics: bool = True


class CreativeResponse(BaseModel):
    """Response body from /creative.

    Attributes:
        request_id: Unique request identifier.
        emotion: Detected emotional tone.
        gated_output: Quality-gated conversational output.
        quality_score: Quality assessment score [0.0, 1.0].
        quality_passed: Whether output passed the quality gate.
        genre: Musical genre for the detected emotion.
        tempo_bpm: Target tempo in beats per minute.
        key: Musical key.
        energy: Energy level [0.0, 1.0].
        valence: Emotional valence [0.0, 1.0].
        music_prompt: Full music generation prompt text.
        style_tags: Comma-separated style tags.
        lyrics: Generated lyrics (empty if include_lyrics=False).
        negative_prompt: Things to avoid in generation.
    """

    request_id: str
    emotion: str
    gated_output: str
    quality_score: float
    quality_passed: bool
    genre: str
    tempo_bpm: int
    key: str
    energy: float
    valence: float
    music_prompt: str
    style_tags: str
    lyrics: str
    negative_prompt: str


# â”€â”€â”€ User Database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_USER_DB_PATH = os.environ.get("CONDROX_USER_DB", "Databases/condrox_users.db")
user_db = UserDatabase(_USER_DB_PATH)

# Seed default admin user if no users exist
if not user_db.user_exists("Dovakina"):
    user_db.create_user("Dovakina", "3f21gsa", role="admin")


# â”€â”€â”€ Auth Request / Response Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class LoginRequest(BaseModel):
    """Request body for /auth/login.

    Attributes:
        username: User's username.
        password: User's password.
    """

    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    """Response body from /auth/login.

    Attributes:
        token: Session token for API authentication.
        username: Authenticated username.
        role: User role.
    """

    token: str
    username: str
    role: str


class VerifyResponse(BaseModel):
    """Response body from /auth/verify.

    Attributes:
        valid: Whether the token is valid.
        user_id: User ID if valid.
    """

    valid: bool
    user_id: int | None = None


# â”€â”€â”€ Application â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Global BrainController instance managed by lifespan
current_controller: BrainController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Boot the BrainController at startup.

    Yields:
        None.
    """
    global current_controller
    current_controller = BrainController()
    current_controller.boot_system()
    # Set static gauges
    PIPELINE_STAGES.set(current_controller.pipeline.stage_count())
    yield
    current_controller = None


app = FastAPI(
    title="Condrox AI API",
    version=CORE_VERSION,
    lifespan=lifespan,
    description="Cognitive conversational intelligence engine with creative stack.",
)

# â”€â”€â”€ Middleware: CORS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Middleware: Local Network Access Control ────────────────────────────────

# Allowed networks: localhost + private LAN (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
# Set CONDROX_ALLOW_EXTERNAL=1 to disable this filter (NOT recommended)
_LOCAL_ONLY: bool = os.environ.get("CONDROX_ALLOW_EXTERNAL", "0") == "0"
_ALLOWED_NETWORKS = {
    "127.0.0.1", "::1", "localhost", "testclient",
}


def _is_local_ip(ip: str) -> bool:
    """Check if IP is localhost or private network range.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    if not ip:
        return True  # Allow empty IP (Starlette TestClient has no client host)
    if ip in _ALLOWED_NETWORKS:
        return True
    # Check private ranges
    if ip.startswith("192.168."):
        return True
    if ip.startswith("10."):
        return True
    if ip.startswith("172."):
        parts = ip.split(".")
        if len(parts) > 1:
            try:
                second = int(parts[1])
                if 16 <= second <= 31:
                    return True
            except ValueError:
                pass
    # IPv6 loopback
    if ip.startswith("fe80:") or ip == "::1":
        return True
    return False


@app.middleware("http")
async def local_network_filter(request: Request, call_next: Any) -> Response:
    """Block all requests from non-local/non-private IP addresses.

    Only allows:
    - 127.0.0.1 / ::1 (localhost)
    - 192.168.x.x (private LAN)
    - 10.x.x.x (private LAN)
    - 172.16-31.x.x (private LAN)

    Set CONDROX_ALLOW_EXTERNAL=1 to disable.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    if not _LOCAL_ONLY:
        return await call_next(request)

    # Get real client IP (check CF/X-Forwarded headers, fallback to request.client)
    client_ip = (
        request.headers.get("CF-Connecting-IP", "")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "")
    )

    if not _is_local_ip(client_ip):
        return Response(
            content=json.dumps({
                "error": "Access denied",
                "detail": "Dashboard API is restricted to local network only.",
            }),
            status_code=403,
            media_type="application/json",
        )

    return await call_next(request)

# â”€â”€â”€ Middleware: Rate Limiting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def rate_limit_counter(request: Request, call_next: Any) -> Response:
    """Count rate-limited requests for Prometheus metrics.

    Complexity: T(n) âˆˆ O(1).
    """
    response = await call_next(request)
    if response.status_code == 429:
        RATE_LIMIT_HITS.inc()
    return response


@app.middleware("http")
async def auth_failure_counter(request: Request, call_next: Any) -> Response:
    """Count authentication failures for Prometheus metrics.

    Complexity: T(n) âˆˆ O(1).
    """
    response = await call_next(request)
    if response.status_code == 401:
        AUTH_FAILURES.inc()
    return response


# â”€â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.post("/auth/login")
@limiter.limit("10/minute")
async def auth_login(request: Request, req: LoginRequest) -> Dict[str, Any]:
    """Authenticate a user and return a session token.

    Args:
        req: LoginRequest with username and password.

    Returns:
        Dict with token, username, and role.

    Raises:
        HTTPException: 401 if credentials are invalid.

    Complexity: T(n) âˆˆ O(i * k) for PBKDF2 verification.
    """
    user = user_db.verify_user(req.username, req.password)
    if user is None:
        # Allow login with static API key as username and any password
        if hmac.compare_digest(req.username, API_KEY):
            AUTH_FAILURES.inc() if False else None
            return {
                "token": API_KEY,
                "username": "admin",
                "role": "admin",
            }
        AUTH_FAILURES.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token = user_db.create_token(user.id)
    return {
        "token": token,
        "username": user.username,
        "role": user.role,
    }


@app.post("/auth/logout")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def auth_logout(request: Request) -> Dict[str, Any]:
    """Revoke a session token.

    Expects the token in the X-API-Key header.

    Returns:
        Dict with logout status.

    Complexity: T(n) âˆˆ O(1).
    """
    token = request.headers.get("X-API-Key", "")
    if token:
        user_db.revoke_token(token)
    return {"message": "Logged out"}


@app.get("/auth/verify")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def auth_verify(request: Request) -> Dict[str, Any]:
    """Verify a session token.

    Expects the token in the X-API-Key header.

    Returns:
        Dict with valid flag and user_id.

    Complexity: T(n) âˆˆ O(1).
    """
    token = request.headers.get("X-API-Key", "")
    user_id = user_db.validate_token(token) if token else None
    return {"valid": user_id is not None, "user_id": user_id}


@app.get("/health")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def health(request: Request) -> Dict[str, Any]:
    """Return aggregate health status.

    Returns:
        Dict with health score and status.

    Complexity: T(n) âˆˆ O(s) for s heartbeat checks.
    """
    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    health_status = current_controller.health()
    HEALTH_SCORE.set(health_status.score)
    return {
        "status": "healthy" if health_status.score >= 0.5 else "unhealthy",
        "score": health_status.score,
    }


@app.get("/status")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def system_status(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Return full system status including all CI components.

    Requires API key authentication.

    Returns:
        Dict with subsystem status.

    Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
    """
    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    snapshot = current_controller.status()
    # Update dynamic gauges
    MEMORY_COUNT.set(snapshot.get("memory_count", 0))
    ACTIVE_GOALS.set(snapshot.get("active_goals", 0))
    TOKEN_USAGE.set(snapshot.get("token_usage", 0))
    PRIORITY_QUEUE_SIZE.set(snapshot.get("priority_queue_size", 0))
    return snapshot


@app.get("/metrics")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def metrics(request: Request) -> Any:
    """Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics text.

    Complexity: T(n) âˆˆ O(m) for m registered metrics.
    """
    return generate_latest(REGISTRY)


@app.post("/process", response_model=ProcessResponse)
@limiter.limit(RATE_LIMIT_PROCESS)
async def process(
    request: Request,
    req: ProcessRequest,
    _: str = Depends(verify_api_key),
) -> ProcessResponse:
    """Process a user request through the Brain pipeline.

    Requires API key authentication.

    Args:
        request: FastAPI request (for rate limiting).
        req: ProcessRequest with input and optional context.

    Returns:
        ProcessResponse with the final BrainState fields.

    Raises:
        HTTPException: 400 for invalid input, 503 if Brain is not ready.

    Complexity: T(n) âˆˆ O(Î£ s_i) for the pipeline.
    """
    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start_time = time.perf_counter()
    try:
        state = current_controller.process_request(req.input, req.context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        elapsed = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(elapsed)
        REQUEST_COUNT.inc()

    # Track safety blocks
    if "safety_blocked" in state.metadata:
        SAFETY_BLOCKS.inc()

    # Record in monitoring tracker
    rt_ms = (time.perf_counter() - start_time) * 1000
    _status = "rejected" if not state.security_approved else (
        "error" if state.errors else "ok"
    )
    _session = req.context.get("session_id", "anonymous") if req.context else "anonymous"
    get_tracker().record(
        session=_session,
        msg=req.input,
        intent=state.intent,
        domain=state.domain,
        mode=state.mode,
        emotion=state.metadata.get("emotion", "neutral"),
        confidence=state.intent_confidence,
        resp=state.final_output,
        errors=state.errors,
        rt_ms=rt_ms,
        status=_status,
    )

    return ProcessResponse(
        request_id=state.request_id,
        final_output=state.final_output,
        mode=state.mode,
        domain=state.domain,
        intent=state.intent,
        intent_confidence=state.intent_confidence,
        security_approved=state.security_approved,
        ethical_approved=state.ethical_approved,
        errors=state.errors,
    )


@app.post("/process/stream")
@limiter.limit(RATE_LIMIT_PROCESS)
async def process_stream(
    request: Request,
    req: ProcessRequest,
    _: str = Depends(verify_api_key),
) -> StreamingResponse:
    """Process a user request through the Brain pipeline with SSE streaming.

    Requires API key authentication.

    Args:
        request: FastAPI request (for rate limiting).
        req: ProcessRequest with input and optional context.

    Returns:
        StreamingResponse with SSE-formatted chunks.

    Raises:
        HTTPException: 400 for invalid input, 503 if Brain is not ready.

    Complexity: T(n) âˆˆ O(Î£ s_i) for the pipeline.
    """
    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start_time = time.perf_counter()
    
    async def generate():
        try:
            # Process the request
            state = current_controller.process_request(req.input, req.context)
            
            # Stream the output in chunks
            chunk_size = 10
            output = state.final_output
            
            for i in range(0, len(output), chunk_size):
                chunk = output[i:i + chunk_size]
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send final metadata
            final_data = {
                "final": True,
                "request_id": state.request_id,
                "mode": state.mode,
                "domain": state.domain,
                "intent": state.intent,
                "intent_confidence": state.intent_confidence,
                "security_approved": state.security_approved,
                "ethical_approved": state.ethical_approved,
                "errors": state.errors,
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except ValueError as exc:
            error_data = {"error": str(exc)}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            elapsed = time.perf_counter() - start_time
            REQUEST_LATENCY.observe(elapsed)
            REQUEST_COUNT.inc()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/creative", response_model=CreativeResponse)
@limiter.limit(RATE_LIMIT_CREATIVE)
async def creative(
    request: Request,
    req: CreativeRequest,
    _: str = Depends(verify_api_key),
) -> CreativeResponse:
    """Generate a creative (music) output package from user input.

    Runs the full pipeline, maps detected emotion to a sound palette,
    builds a music generation prompt, and gates the conversational output
    through the quality guardrail.

    Requires API key authentication.

    Args:
        request: FastAPI request (for rate limiting).
        req: CreativeRequest with input and lyrics flag.

    Returns:
        CreativeResponse with emotion, palette, quality, and music prompt.

    Raises:
        HTTPException: 400 for invalid input, 503 if Brain is not ready.

    Complexity: T(n) âˆˆ O(pipeline + palette + prompt).
    """
    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start_time = time.perf_counter()
    try:
        result = current_controller.generate_creative_output(
            req.input, include_lyrics=req.include_lyrics
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        elapsed = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(elapsed)
        REQUEST_COUNT.inc()

    state = result["state"]
    palette = result["palette"]
    quality = result["quality"]
    music_prompt = result["music_prompt"]

    return CreativeResponse(
        request_id=state.request_id,
        emotion=state.emotion,
        gated_output=result["gated_output"],
        quality_score=quality.score,
        quality_passed=quality.passed,
        genre=palette.genre,
        tempo_bpm=palette.tempo_bpm,
        key=palette.key,
        energy=palette.energy,
        valence=palette.valence,
        music_prompt=music_prompt.prompt_text,
        style_tags=music_prompt.style_tags,
        lyrics=music_prompt.lyrics,
        negative_prompt=music_prompt.negative_prompt,
    )


# â”€â”€â”€ AI-OS Endpoints (Phase 13) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@app.get("/aios/status")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_status(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get full AI-OS kernel status.

    Returns boot info, services, health, monitor, task queue, self-repair,
    hot-reloader, data flow, and state manager status.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(s + e) for s services, e events.
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")
    return current_controller.aios_kernel.get_status()


@app.get("/aios/health")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_health(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get AI-OS health snapshot.

    Returns overall health score, alert count, and degraded services.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(1).
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")
    health = current_controller.aios_kernel.get_health()
    if health is None:
        return {"overall_score": 1.0, "alert_count": 0, "degraded_services": []}
    return {
        "overall_score": health.overall_score,
        "alert_count": health.alert_count,
        "degraded_services": [
            n for n, (h, _) in health.service_health.items() if not h
        ],
        "timestamp": health.timestamp,
    }


@app.get("/aios/events")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_events(
    request: Request,
    event_type: Optional[str] = None,
    limit: int = 100,
    severity: Optional[str] = None,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get AI-OS event log.

    Args:
        event_type: Optional event type filter.
        limit: Maximum events (default 100, max 1000).
        severity: Optional severity filter.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(n) for n events.
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")

    from core.os.event_bus import EventType as ET

    et = None
    if event_type:
        try:
            et = ET(event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event_type. Valid types: {[e.value for e in ET]}",
            )

    limit = min(max(limit, 1), 1000)
    events = current_controller.aios_kernel.get_events(
        event_type=et, limit=limit, severity=severity
    )
    return {
        "events": [
            {
                "type": e.event_type.value,
                "source": e.source,
                "message": e.message,
                "severity": e.severity,
                "timestamp": e.timestamp,
                "data": e.data,
            }
            for e in events
        ],
        "count": len(events),
    }


@app.get("/aios/services")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_services(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get AI-OS service manager status.

    Returns all registered services with their states.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(s) for s services.
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")
    services = current_controller.aios_kernel.service_manager.get_status()
    return {"services": services, "count": len(services)}


@app.get("/aios/task-queue")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_task_queue(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get AI-OS task queue status.

    Returns queue state, pending/running/completed counts, and recent tasks.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(1).
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")
    return current_controller.aios_kernel.task_queue.get_status()


@app.get("/aios/data-flow")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_data_flow(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get AI-OS data flow manager status.

    Returns channels, routing rules, and per-channel metrics.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(c) for c channels.
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")
    return current_controller.aios_kernel.data_flow.get_status()


@app.post("/aios/shutdown")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def aios_shutdown(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Gracefully shut down the AI-OS kernel.

    Stops monitoring, task queue, hot-reloader, data flow, and services.
    Saves final state checkpoint.

    Requires API key authentication.

    Complexity: T(n) âˆˆ O(Î£ s_i) for service stops.
    """
    if current_controller is None or current_controller.aios_kernel is None:
        raise HTTPException(status_code=503, detail="AI-OS not initialized")
    if not current_controller.aios_kernel.is_booted:
        raise HTTPException(status_code=400, detail="AI-OS not booted")
    duration = current_controller.aios_kernel.shutdown()
    return {"shutdown_time": round(duration, 3), "status": "shutdown complete"}


# â”€â”€â”€ Admin Router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.include_router(admin_router, dependencies=[Depends(verify_api_key)])


# Monitoring Endpoints (Pre-Release Monitor + Live Traffic) ─────────────

class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


@app.get("/monitoring/live")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def monitoring_live(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get live traffic data for the dashboard.

    Returns recent request feed, total counts, active sessions,
    and country distribution.

    Complexity: T(n) O(k) for k records, S(n) O(k).
    """
    return get_tracker().get_live()


@app.get("/monitoring/stats")
@limiter.limit(RATE_LIMIT_DEFAULT)
async def monitoring_stats(
    request: Request,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get aggregate monitoring stats for the dashboard.

    Returns total counts, error rate, intent/domain/mode/emotion
    distributions, response time stats, and recent messages.

    Complexity: T(n) O(k) for k records, S(n) O(e).
    """
    return get_tracker().get_stats()


@app.post("/chat")
@limiter.limit(RATE_LIMIT_PROCESS)
async def chat(
    request: Request,
    req: ChatRequest,
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Simple chat endpoint for the Pre-Release Monitor chat widget.

    Processes a message through the Brain pipeline and returns
    response, intent, and response time.

    Complexity: T(n) O(sum s_i) for the pipeline.
    """
    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    start_time = time.perf_counter()
    try:
        state = current_controller.process_request(req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rt_ms = (time.perf_counter() - start_time) * 1000
    _session = req.session_id or f"chat-{int(time.time())}"
    _status = "rejected" if not state.security_approved else (
        "error" if state.errors else "ok"
    )
    get_tracker().record(
        session=_session,
        msg=req.message,
        intent=state.intent,
        domain=state.domain,
        mode=state.mode,
        emotion=state.metadata.get("emotion", "neutral"),
        confidence=state.intent_confidence,
        resp=state.final_output,
        errors=state.errors,
        rt_ms=rt_ms,
        status=_status,
    )

    return {
        "response": state.final_output,
        "intent": state.intent,
        "domain": state.domain,
        "mode": state.mode,
        "emotion": state.metadata.get("emotion", "neutral"),
        "confidence": state.intent_confidence,
        "session_id": _session,
        "response_time_ms": round(rt_ms, 1),
        "errors": state.errors,
    }


# â”€â”€â”€ Main Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# v3.4.0 PATCH: Added __main__ block so `python -m api.main` starts the server
# with proper host/port binding. Previously only `uvicorn api.main:app` worked,
# which caused "API not reachable" warnings in evidence runs when the server
# wasn't started separately.

if __name__ == "__main__":
    import uvicorn

    _host = os.environ.get("CONDROX_API_HOST", "0.0.0.0")
    _port = int(os.environ.get("CONDROX_API_PORT", "8000"))
    _reload = os.environ.get("CONDROX_API_RELOAD", "0") == "1"

    print(f"Condrox AI API server starting on {_host}:{_port}")
    print(f"  Health endpoint: http://localhost:{_port}/health")
    print(f"  API docs:        http://localhost:{_port}/docs")
    print(f"  Metrics:         http://localhost:{_port}/metrics")
    print(f"  Reload:          {_reload}")

    uvicorn.run(
        "api.main:app",
        host=_host,
        port=_port,
        reload=_reload,
    )
