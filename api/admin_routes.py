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
Admin API router for management dashboard endpoints.

Provides CRUD operations for memory and knowledge stores, pipeline tracing,
state persistence control, agent inspection, and configuration viewing.

All endpoints require API key authentication.

Mathematical Foundation:
- Memory/Knowledge listing: O(n) for n items.
- Delete operations: O(1) per item.
- State save/load: O(n) for n total items.
"""

from __future__ import annotations

import hashlib as _hashlib
import json
import mimetypes
import os
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.brain.brain_controller import BrainController

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
)


def _get_controller() -> BrainController:
    """Get the global BrainController instance.

    Returns:
        The active BrainController.

    Raises:
        HTTPException: 503 if Brain is not initialized.
    """
    from api.main import current_controller

    if current_controller is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return current_controller


# ─── Audit Log ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Immutable audit log entry.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    timestamp: float
    action: str
    target: str
    details: str
    result: str
    api_key_id: str
    ip_address: str


_audit_log: List[AuditEntry] = []
_AUDIT_MAX = 1000


def _log_audit(request: Request, action: str, target: str, details: str, result: str = "success") -> None:
    """Append an entry to the in-memory audit log.

    Complexity: T(n) ∈ O(1) amortized (deque-like trim), S(n) ∈ O(1).
    """
    api_key = request.headers.get("X-API-Key", "unknown")
    ip = request.client.host if request.client else "unknown"
    entry = AuditEntry(
        timestamp=_time.time(),
        action=action,
        target=target,
        details=details[:200],
        result=result,
        api_key_id=api_key[:8] if api_key else "unknown",
        ip_address=ip,
    )
    _audit_log.append(entry)
    if len(_audit_log) > _AUDIT_MAX:
        del _audit_log[:len(_audit_log) - _AUDIT_MAX]


# ─── Request / Response Models ───────────────────────────────────────────────


class MemoryItemResponse(BaseModel):
    """Response model for a single memory item."""

    item_id: str
    content: str
    source: str
    tags: List[str]
    created_at: float
    access_count: int
    weight: float


class KnowledgeItemResponse(BaseModel):
    """Response model for a single knowledge item."""

    item_id: str
    domain: str
    content: str
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestKnowledgeRequest(BaseModel):
    """Request body for ingesting knowledge."""

    content: str = Field(..., min_length=1, max_length=8192)
    domain: str = Field(default="general")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AddMemoryRequest(BaseModel):
    """Request body for adding a memory item."""

    content: str = Field(..., min_length=1, max_length=4096)
    source: str = Field(default="manual")
    tags: List[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class SearchMemoryRequest(BaseModel):
    """Request body for searching memories."""

    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=100)


class UpdateConfigRequest(BaseModel):
    """Request body for updating configuration fields."""

    blocked_patterns: Optional[List[str]] = None
    blocked_topics: Optional[List[str]] = None
    max_input_length: Optional[int] = Field(default=None, ge=1, le=65536)
    default_profile: Optional[str] = None
    log_level: Optional[str] = None
    enable_learning: Optional[bool] = None


class SearchKnowledgeRequest(BaseModel):
    """Request body for searching knowledge."""

    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=100)
    domain_filter: Optional[str] = None


class PipelineTraceResponse(BaseModel):
    """Response model for pipeline trace."""

    request_id: str
    stages: List[Dict[str, Any]]
    final_output: str
    mode: str
    domain: str
    intent: str
    emotion: str
    errors: List[str]


class StateSaveResponse(BaseModel):
    """Response model for state save."""

    items_saved: int
    message: str


class AgentInfoResponse(BaseModel):
    """Response model for agent info."""

    agent_id: str
    mode: str
    status: str


# ─── Admin Endpoints ─────────────────────────────────────────────────────────


@router.get("/health")
async def admin_health() -> Dict[str, Any]:
    """Detailed health check with all heartbeat subsystems.

    Returns:
        Dict with aggregate health and per-subsystem status.

    Complexity: T(n) ∈ O(s) for s heartbeat checks.
    """
    controller = _get_controller()
    health = controller.health()
    return {
        "aggregate_score": health.score,
        "subsystem": health.subsystem,
        "message": health.message,
        "timestamp": health.timestamp,
    }


@router.get("/config")
async def get_config(request: Request) -> Dict[str, Any]:
    """Return the current BrainController configuration.

    Returns:
        Dict with configuration fields.

    Complexity: T(n) ∈ O(1).
    """
    controller = _get_controller()
    config = controller.config
    _log_audit(request, "config_view", "config", "Viewed all config fields")
    return {
        "system_name": config.system_name,
        "version": config.version,
        "environment": config.environment,
        "log_level": config.log_level,
        "max_input_length": config.max_input_length,
        "default_profile": config.default_profile,
        "enable_learning": config.enable_learning,
        "enable_single_instance_lock": config.enable_single_instance_lock,
        "persistence_path": config.persistence_path,
        "memory_decay_halflife_seconds": config.memory_decay_halflife_seconds,
        "pipeline_stages": controller.pipeline.stage_count(),
        "blocked_patterns": list(config.blocked_patterns),
        "blocked_topics": list(config.blocked_topics),
    }


@router.put("/config")
async def update_config(req: UpdateConfigRequest, request: Request) -> Dict[str, Any]:
    """Update editable configuration fields.

    Args:
        req: Fields to update (only non-None fields are applied).

    Returns:
        Dict with updated fields and confirmation.

    Complexity: T(n) ∈ O(1).
    """
    controller = _get_controller()
    config = controller.config
    updated: List[str] = []

    if req.blocked_patterns is not None:
        object.__setattr__(config, "blocked_patterns", tuple(req.blocked_patterns))
        controller.security_engine.blocked_patterns = [
            __import__("re").compile(p) for p in req.blocked_patterns
        ]
        updated.append("blocked_patterns")
    if req.blocked_topics is not None:
        object.__setattr__(config, "blocked_topics", tuple(req.blocked_topics))
        controller.ethics_engine.blocked_topics = [t.lower() for t in req.blocked_topics]
        updated.append("blocked_topics")
    if req.max_input_length is not None:
        object.__setattr__(config, "max_input_length", req.max_input_length)
        updated.append("max_input_length")
    if req.default_profile is not None:
        object.__setattr__(config, "default_profile", req.default_profile)
        updated.append("default_profile")
    if req.log_level is not None:
        object.__setattr__(config, "log_level", req.log_level)
        updated.append("log_level")
    if req.enable_learning is not None:
        object.__setattr__(config, "enable_learning", req.enable_learning)
        updated.append("enable_learning")

    _log_audit(request, "config_edit", "config", f"Updated fields: {', '.join(updated)}")
    return {
        "message": f"Updated {len(updated)} field(s)",
        "updated_fields": updated,
        "blocked_patterns": list(config.blocked_patterns),
        "blocked_topics": list(config.blocked_topics),
        "max_input_length": config.max_input_length,
        "default_profile": config.default_profile,
        "log_level": config.log_level,
        "enable_learning": config.enable_learning,
    }


@router.get("/memory")
async def list_memory() -> Dict[str, Any]:
    """List all stored memory items.

    Returns:
        Dict with list of memory items and count.

    Complexity: T(n) ∈ O(n) for n memory items.
    """
    controller = _get_controller()
    items = controller.memory_manager.store.all_items()
    return {
        "count": len(items),
        "items": [
            {
                "item_id": item.item_id,
                "content": item.content,
                "source": item.source,
                "tags": list(item.tags),
                "created_at": item.created_at,
                "access_count": item.access_count,
                "weight": item.weight,
            }
            for item in items
        ],
    }


@router.post("/memory")
async def add_memory(req: AddMemoryRequest, request: Request) -> Dict[str, Any]:
    """Add a new memory item.

    Returns:
        Dict with the new item_id.

    Complexity: T(n) ∈ O(n + d²) for content length d.
    """
    controller = _get_controller()
    item_id = controller.add_memory(
        content=req.content,
        source=req.source,
        tags=req.tags if req.tags else None,
        weight=req.weight,
    )
    _log_audit(request, "memory_create", item_id, f"Added memory: {req.content[:100]}")
    return {"item_id": item_id, "message": "Memory added"}


@router.post("/memory/search")
async def search_memory(req: SearchMemoryRequest) -> Dict[str, Any]:
    """Search memories by semantic similarity.

    Returns:
        Dict with matching items.

    Complexity: T(n) ∈ O(n log k) for n items, top_k=k.
    """
    controller = _get_controller()
    items = controller.memory_manager.search(query=req.query, top_k=req.top_k)
    return {
        "count": len(items),
        "items": [
            {
                "item_id": item.item_id,
                "content": item.content,
                "source": item.source,
                "tags": list(item.tags),
                "weight": item.weight,
                "access_count": item.access_count,
            }
            for item in items
        ],
    }


@router.delete("/memory/{item_id}")
async def delete_memory(item_id: str, request: Request) -> Dict[str, Any]:
    """Delete a specific memory item by ID.

    Returns:
        Dict with deletion status.

    Complexity: T(n) ∈ O(1).
    """
    controller = _get_controller()
    if item_id not in controller.memory_manager.store._items:
        raise HTTPException(status_code=404, detail=f"Memory item {item_id} not found")
    del controller.memory_manager.store._items[item_id]
    _log_audit(request, "memory_delete", item_id, f"Deleted memory item {item_id}")
    return {"item_id": item_id, "message": "Memory deleted"}


@router.delete("/memory")
async def clear_all_memory() -> Dict[str, Any]:
    """Clear all memory items.

    Returns:
        Dict with count of cleared items.

    Complexity: T(n) ∈ O(n) for n items.
    """
    controller = _get_controller()
    count = controller.memory_manager.store.count()
    controller.memory_manager.store._items.clear()
    return {"cleared": count, "message": "All memory cleared"}


@router.get("/knowledge")
async def list_knowledge() -> Dict[str, Any]:
    """List all stored knowledge items.

    Returns:
        Dict with list of knowledge items and count.

    Complexity: T(n) ∈ O(n) for n knowledge items.
    """
    controller = _get_controller()
    backend = controller.knowledge_router.backend
    items = list(backend._items.values())
    return {
        "count": len(items),
        "items": [
            {
                "item_id": item.item_id,
                "domain": item.domain,
                "content": item.content,
                "confidence": item.confidence,
                "metadata": dict(item.metadata) if item.metadata else {},
            }
            for item in items
        ],
    }


@router.post("/knowledge")
async def ingest_knowledge(req: IngestKnowledgeRequest, request: Request) -> Dict[str, Any]:
    """Ingest new knowledge into the backend.

    Returns:
        Dict with the new item_id.

    Complexity: T(n) ∈ O(n + d²) for content length d.
    """
    controller = _get_controller()
    item_id = controller.ingest_knowledge(content=req.content, domain=req.domain)
    _log_audit(request, "knowledge_ingest", item_id, f"Ingested: {req.content[:100]} (domain={req.domain})")
    return {"item_id": item_id, "message": "Knowledge ingested"}


@router.post("/knowledge/search")
async def search_knowledge(req: SearchKnowledgeRequest) -> Dict[str, Any]:
    """Search knowledge by semantic similarity.

    Returns:
        Dict with matching items.

    Complexity: T(n) ∈ O(n log k) for n items, top_k=k.
    """
    controller = _get_controller()
    items = controller.knowledge_router.backend.search(
        query=req.query, top_k=req.top_k, domain_filter=req.domain_filter
    )
    return {
        "count": len(items),
        "items": [
            {
                "item_id": item.item_id,
                "domain": item.domain,
                "content": item.content,
                "confidence": item.confidence,
            }
            for item in items
        ],
    }


@router.delete("/knowledge/{item_id}")
async def delete_knowledge(item_id: str, request: Request) -> Dict[str, Any]:
    """Delete a specific knowledge item by ID.

    Returns:
        Dict with deletion status.

    Complexity: T(n) ∈ O(1).
    """
    controller = _get_controller()
    backend = controller.knowledge_router.backend
    if item_id not in backend._items:
        raise HTTPException(status_code=404, detail=f"Knowledge item {item_id} not found")
    # Remove from domain index
    item = backend._items[item_id]
    domain_set = backend._domain_index.get(item.domain)
    if domain_set is not None:
        domain_set.discard(item_id)
        if not domain_set:
            del backend._domain_index[item.domain]
    del backend._items[item_id]
    _log_audit(request, "knowledge_delete", item_id, f"Deleted knowledge item {item_id}")
    return {"item_id": item_id, "message": "Knowledge deleted"}


@router.delete("/knowledge")
async def clear_all_knowledge() -> Dict[str, Any]:
    """Clear all knowledge items.

    Returns:
        Dict with count of cleared items.

    Complexity: T(n) ∈ O(n) for n items.
    """
    controller = _get_controller()
    backend = controller.knowledge_router.backend
    count = backend.count()
    backend._items.clear()
    backend._domain_index.clear()
    return {"cleared": count, "message": "All knowledge cleared"}


@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents and their status.

    Returns:
        Dict with agent list.

    Complexity: T(n) ∈ O(a) for a agents.
    """
    controller = _get_controller()
    return {
        "count": controller.agent_registry.count(),
        "modes": list(controller.agent_registry.modes()),
        "agents": controller.agent_registry.status(),
    }


@router.get("/pipeline")
async def pipeline_info() -> Dict[str, Any]:
    """Return pipeline stage information.

    Returns:
        Dict with stage count and names.

    Complexity: T(n) ∈ O(s) for s stages.
    """
    controller = _get_controller()
    stages = []
    for stage in controller.pipeline._stages:
        name = type(stage).__name__
        stages.append({"name": name, "class": type(stage).__module__})
    return {
        "stage_count": controller.pipeline.stage_count(),
        "stages": stages,
    }


@router.get("/pipeline/telemetry")
async def pipeline_telemetry() -> Dict[str, Any]:
    """Return per-stage aggregated metrics from the PipelineMonitor.

    Returns:
        Dict with total_requests, per-stage metrics (avg/min/max duration,
        success rate, execution counts), and total spans.

    Complexity: T(n) ∈ O(s) for s stages.
    """
    controller = _get_controller()
    return controller.pipeline_monitor.export_metrics()


@router.get("/pipeline/spans")
async def pipeline_spans(limit: int = 50) -> Dict[str, Any]:
    """Return recent pipeline execution spans.

    Args:
        limit: Maximum spans to return (default 50, max 500).

    Returns:
        Dict with span list and count.

    Complexity: T(n) ∈ O(n) for n spans.
    """
    controller = _get_controller()
    spans = controller.pipeline_monitor.get_spans(limit=min(limit, 500))
    return {
        "count": len(spans),
        "spans": [
            {
                "stage_name": s.stage_name,
                "request_id": s.request_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_ms": round(s.duration_ms, 3),
                "success": s.success,
                "error": s.error,
            }
            for s in spans
        ],
    }


@router.post("/state/save", response_model=StateSaveResponse)
async def save_state(request: Request) -> StateSaveResponse:
    """Persist memory and knowledge to storage backend.

    Returns:
        StateSaveResponse with count of saved items.

    Complexity: T(n) ∈ O(n) for n total items.
    """
    controller = _get_controller()
    saved = controller.save_state()
    _log_audit(request, "state_save", "state", f"Saved {saved} items")
    return StateSaveResponse(
        items_saved=saved,
        message=f"Saved {saved} items to persistence backend"
        if saved > 0
        else "No persistence backend configured",
    )


@router.post("/state/load")
async def load_state(request: Request) -> Dict[str, Any]:
    """Load memory and knowledge from storage backend.

    Returns:
        Dict with loaded counts.

    Complexity: T(n) ∈ O(n) for n stored items.
    """
    controller = _get_controller()
    if controller._storage_backend is None:
        raise HTTPException(status_code=400, detail="No persistence backend configured")
    mem_count = controller.memory_manager.store.load(
        controller._storage_backend, table="memories"
    )
    know_count = controller.knowledge_router.backend.load(
        controller._storage_backend, table="knowledge"
    )
    _log_audit(request, "state_load", "state", f"Loaded {mem_count} memories, {know_count} knowledge")
    return {
        "memories_loaded": mem_count,
        "knowledge_loaded": know_count,
        "message": f"Loaded {mem_count} memories and {know_count} knowledge items",
    }


def _get_knowledge_path() -> str:
    """Resolve the knowledge store path from env or default."""
    return os.environ.get(
        "KNOWLEDGE_DATA_PATH",
        r"data/knowledge",
    )


def _file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file (first 1MB only for speed on large files).

    Complexity: T(n) ∈ O(min(n, 1MB)) where n is file size.
    """
    hasher = _hashlib.sha256()
    bytes_read = 0
    max_bytes = 1024 * 1024  # 1MB cap for hash
    with open(path, "rb") as f:
        while bytes_read < max_bytes:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            bytes_read += len(chunk)
    return hasher.hexdigest()[:16]


def _mime_type(path: Path) -> str:
    """Guess MIME type from file extension."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


@router.get("/knowledge-store")
async def list_knowledge_store() -> Dict[str, Any]:
    """List knowledge store domains and files on disk (DFIL — Domain-First Indexing Layer).

    Scans the knowledge directory and returns per-domain file listings with
    full metadata: hash, MIME type, size, timestamps, and file type aggregation.
    No content ingestion — metadata only.

    Complexity: T(n) ∈ O(n) for n files (hash capped at 1MB per file).
    """
    base_path = _get_knowledge_path()
    base = Path(base_path)

    if not base.exists():
        return {
            "path": base_path,
            "exists": False,
            "domains": [],
            "total_files": 0,
            "total_size_mb": 0,
            "scanned_at": _time.time(),
        }

    domains: List[Dict[str, Any]] = []
    total_files = 0
    total_size = 0
    all_file_types: Dict[str, int] = {}

    for domain_dir in sorted(base.iterdir()):
        if not domain_dir.is_dir():
            continue

        files: List[Dict[str, Any]] = []
        domain_size = 0
        domain_types: Dict[str, int] = {}
        meta: Dict[str, Any] = {}
        subdirs: List[str] = []

        # Recursive scan — rglob finds files in all subdirectories
        for entry in sorted(domain_dir.rglob("*")):
            if entry.is_dir():
                subdirs.append(entry.name)
                continue
            if not entry.is_file():
                continue

            stat = entry.stat()
            size = stat.st_size
            domain_size += size
            total_size += size
            total_files += 1

            ext = entry.suffix.lower() or "(no ext)"
            mime = _mime_type(entry)
            domain_types[ext] = domain_types.get(ext, 0) + 1
            all_file_types[ext] = all_file_types.get(ext, 0) + 1

            file_info: Dict[str, Any] = {
                "name": entry.name,
                "path": str(entry.relative_to(base)),
                "extension": ext,
                "mime_type": mime,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
                "hash": _file_hash(entry) if size < 100 * 1024 * 1024 else None,  # Skip hash for files > 100MB
            }

            # Read meta.json if present
            if entry.name == "meta.json":
                try:
                    meta = json.loads(entry.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Read first 200 chars of small .json data files for preview
            if entry.suffix == ".json" and entry.name != "meta.json" and size < 1024 * 1024:
                try:
                    file_info["preview"] = entry.read_text(encoding="utf-8", errors="replace")[:200]
                except Exception:
                    file_info["preview"] = ""

            files.append(file_info)

        domains.append({
            "name": domain_dir.name,
            "file_count": len(files),
            "size_mb": round(domain_size / (1024 * 1024), 2),
            "size_gb": round(domain_size / (1024 * 1024 * 1024), 2),
            "file_types": domain_types,
            "subdirs": subdirs,
            "files": files,
            "meta": meta,
        })

    return {
        "path": base_path,
        "exists": True,
        "domains": domains,
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
        "all_file_types": all_file_types,
        "domain_count": len(domains),
        "scanned_at": _time.time(),
    }


@router.get("/knowledge-store/{domain}")
async def list_knowledge_store_domain(domain: str) -> Dict[str, Any]:
    """Return files for a specific knowledge domain.

    Args:
        domain: Domain directory name.

    Returns:
        Dict with file list and count.

    Complexity: T(n) ∈ O(n) for n files in domain.
    """
    base = Path(_get_knowledge_path())
    domain_dir = base / domain
    if not domain_dir.is_dir():
        return {"domain": domain, "files": [], "count": 0}

    files: List[Dict[str, Any]] = []
    for entry in sorted(domain_dir.rglob("*")):
        if not entry.is_file():
            continue
        stat = entry.stat()
        ext = entry.suffix.lower().lstrip(".") or "unknown"
        mime = _mime_type(entry)
        file_hash = ""
        if stat.st_size < 100 * 1024 * 1024:
            try:
                import hashlib
                h = hashlib.sha256()
                with entry.open("rb") as f:
                    h.update(f.read(1024 * 1024))
                file_hash = h.hexdigest()[:16]
            except Exception:
                pass
        files.append({
            "filename": entry.name,
            "filetype": ext,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 4),
            "path": str(entry.relative_to(base)),
            "hash": file_hash,
            "created": _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(stat.st_ctime)),
            "modified": _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(stat.st_mtime)),
            "mime_type": mime,
        })

    return {"domain": domain, "files": files, "count": len(files)}


@router.get("/security/events")
async def list_security_events(
    event_type: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """List security and ethics events from the safety layer audit log.

    Args:
        event_type: Filter by type (pii, secret, injection, ethics).
        action: Filter by action (blocked, warning, approved).
        limit: Maximum entries (default 100, max 500).

    Returns:
        Dict with events list, counts by type, and total count.

    Complexity: T(n) ∈ O(n) for n audit log entries.
    """
    controller = _get_controller()
    limit = min(max(1, limit), 500)

    violations = controller.safety_layer.get_audit_log(limit=limit)

    # Map rule_id to event type
    _TYPE_MAP = {
        "pii": "pii",
        "secret": "secret",
        "injection": "injection",
        "ethics": "ethics",
        "blocked_topic": "ethics",
        "blocked_pattern": "injection",
    }

    def _classify(v) -> str:
        rid = v.rule_id.lower()
        for key, val in _TYPE_MAP.items():
            if key in rid:
                return val
        return "injection"

    def _action(v) -> str:
        level = str(v.level).upper()
        if "BLOCK" in level or "CRITICAL" in level:
            return "blocked"
        if "WARN" in level:
            return "warning"
        return "approved"

    events = []
    for v in violations:
        evt_type = _classify(v)
        evt_action = _action(v)
        if event_type and evt_type != event_type:
            continue
        if action and evt_action != action:
            continue
        events.append({
            "id": f"{v.rule_id}_{v.timestamp}",
            "timestamp": v.timestamp,
            "type": evt_type,
            "action": evt_action,
            "request_id": "",
            "details": v.message,
            "pattern": v.matched_text[:100] if v.matched_text else None,
        })

    # Counts by type
    counts = {"pii": 0, "secret": 0, "injection": 0, "ethics": 0}
    for v in violations:
        counts[_classify(v)] = counts.get(_classify(v), 0) + 1

    return {
        "count": len(events),
        "events": events,
        "stats": counts,
    }


@router.get("/errors")
async def list_errors(
    severity: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """List pipeline errors from the error recovery layer.

    Args:
        severity: Filter by severity (CRITICAL, WARNING, INFO).
        stage: Filter by stage name.
        limit: Maximum entries (default 100, max 500).

    Returns:
        Dict with error list, counts by severity, and total count.

    Complexity: T(n) ∈ O(n) for n error log entries.
    """
    controller = _get_controller()
    limit = min(max(1, limit), 500)

    error_records = controller.error_recovery.get_error_log(limit=limit)

    def _severity(rec) -> str:
        if rec.recovery_result and not rec.recovery_result.success:
            return "CRITICAL"
        if rec.recovery_attempted:
            return "WARNING"
        return "INFO"

    def _recovery_status(rec) -> str:
        if not rec.recovery_attempted:
            return "none"
        if rec.recovery_result is None:
            return "retrying"
        if rec.recovery_result.success:
            return "recovered"
        return "failed"

    errors = []
    for rec in error_records:
        sev = _severity(rec)
        if severity and sev != severity:
            continue
        if stage and rec.stage_name != stage:
            continue
        errors.append({
            "id": rec.error_id,
            "timestamp": rec.timestamp,
            "severity": sev,
            "stage": rec.stage_name,
            "exception": rec.error_message,
            "request_id": rec.error_id,
            "stack_trace": None,
            "recovery_status": _recovery_status(rec),
            "retry_count": controller.error_recovery._retry_counts.get(rec.error_id, 0),
        })

    # Counts by severity
    counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
    for rec in error_records:
        counts[_severity(rec)] = counts.get(_severity(rec), 0) + 1

    return {
        "count": len(errors),
        "errors": errors,
        "stats": counts,
        "recovered": sum(1 for rec in error_records if rec.recovery_result and rec.recovery_result.success),
    }


@router.get("/audit")
async def list_audit_log(
    action_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """List audit log entries with optional filtering.

    Args:
        action_filter: Filter by action type (e.g. memory_create, knowledge_delete).
        limit: Maximum entries to return (default 100, max 500).
        offset: Pagination offset.

    Returns:
        Dict with audit entries and total count.

    Complexity: T(n) ∈ O(n) for n total audit entries (filter scan).
    """
    limit = min(max(1, limit), 500)
    offset = max(0, offset)

    entries = _audit_log
    if action_filter:
        entries = [e for e in entries if action_filter in e.action]

    total = len(entries)
    paged = entries[offset:offset + limit]

    return {
        "count": total,
        "entries": [
            {
                "timestamp": e.timestamp,
                "action": e.action,
                "target": e.target,
                "details": e.details,
                "result": e.result,
                "api_key_id": e.api_key_id,
                "ip_address": e.ip_address,
            }
            for e in reversed(paged)  # Newest first
        ],
    }


# ─── Admin Agent ──────────────────────────────────────────────────────────────


class AdminAgentRequest(BaseModel):
    """Request body for admin agent."""
    message: str = Field(..., min_length=1, max_length=500, description="Natural language admin command")


@router.post("/agent")
async def admin_agent(req: AdminAgentRequest, request: Request) -> Dict[str, Any]:
    """Process a natural language admin command.

    Routes the message through the AdminAgent which can manage
    knowledge, memory, pipeline, state, and system configuration.

    SECURITY: This endpoint is only accessible through the authenticated
    admin API. It is NEVER exposed on the public pre-release API.

    Args:
        req: AdminAgentRequest with the command message.

    Returns:
        Dict with response, command type, success flag, and data.

    Complexity: T(n) in O(command_dispatch), S(n) in O(response).
    """
    controller = _get_controller()
    from agents.AdminAgent import AdminAgent

    agent = AdminAgent(controller)
    result = await agent.process(req.message)

    _log_audit(
        request,
        action="admin_agent",
        target=result.command,
        details=req.message[:100],
        result="success" if result.success else "error",
    )

    return {
        "response": result.response,
        "command": result.command,
        "success": result.success,
        "data": result.data,
        "execution_time_ms": result.execution_time_ms,
    }


# ─── Rate Limits ─────────────────────────────────────────────────────────────


@router.get("/rate-limits")
async def get_rate_limits(request: Request) -> Dict[str, Any]:
    """Return current rate limit configuration for all endpoints.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    import os
    default_limit = os.environ.get("CONDROX_RATE_LIMIT", "100/minute")
    process_limit = os.environ.get("CONDROX_RATE_LIMIT_PROCESS", "30/minute")
    creative_limit = os.environ.get("CONDROX_RATE_LIMIT_CREATIVE", "10/minute")

    def parse_limit(s: str) -> int:
        try:
            return int(s.split("/")[0])
        except (ValueError, IndexError):
            return 100

    limits = [
        {
            "endpoint": "/process",
            "limit": parse_limit(process_limit),
            "current": 0,
            "remaining": parse_limit(process_limit),
            "window": "minute",
        },
        {
            "endpoint": "/creative",
            "limit": parse_limit(creative_limit),
            "current": 0,
            "remaining": parse_limit(creative_limit),
            "window": "minute",
        },
        {
            "endpoint": "/admin/*",
            "limit": parse_limit(default_limit),
            "current": 0,
            "remaining": parse_limit(default_limit),
            "window": "minute",
        },
        {
            "endpoint": "/auth/login",
            "limit": 10,
            "current": 0,
            "remaining": 10,
            "window": "minute",
        },
    ]
    return {"limits": limits, "total": len(limits)}
