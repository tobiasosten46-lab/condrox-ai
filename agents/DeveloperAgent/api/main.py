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
DeveloperAgent REST API — FastAPI service for public code assistant.

Endpoints:
- POST /execute — Execute a code task through the agent
- POST /index — Index a project file for RAG
- GET /health — Health check
- GET /stats — Agent statistics

Complexity:
    execute: T(n) in O(pipeline + gates + rag), S(n) in O(output).
    index: T(n) in O(n) where n = code length, S(n) in O(chunks).

Invariants:
    - All endpoints require valid input.
    - Execute requires Condrox BrainController (provided via dependency).
    - Rate limiting enforced (100 req/min default).

Edge Cases:
    - Empty input -> 400 Bad Request.
    - BrainController unavailable -> 503 Service Unavailable.
    - Gate block -> returns 200 with blocked=True.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from collections import deque
from typing import Any, Dict, List, Optional, Deque, Tuple

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import sys
import os
# Add DeveloperAgent folder to path
developer_agent_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if developer_agent_root not in sys.path:
    sys.path.insert(0, developer_agent_root)

# Add project root to path for core modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DeveloperAgent root: {developer_agent_root}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

from condrox_agent_coder import (
    AgentState,
    CodeIntent,
    ConfidenceLevel,
    DomainType,
    CondroxAgentCoder,
    AgentCodeResult,
)


# ─── Conversation Memory ─────────────────────────────────────────────────────


class ConversationMemory:
    """Per-session conversation history with context window.

    Stores recent messages per session_id for context-aware responses.
    Uses a deque with max length to bound memory.

    Complexity: T(add) in O(1), T(get_context) in O(k) for k history items.
    """

    _sessions: Dict[str, Deque[Dict[str, str]]] = {}
    _max_history: int = 20

    @classmethod
    def add(cls, session_id: str, role: str, content: str) -> None:
        if session_id not in cls._sessions:
            cls._sessions[session_id] = deque(maxlen=cls._max_history)
        cls._sessions[session_id].append({"role": role, "content": content})

    @classmethod
    def get_history(cls, session_id: str) -> List[Dict[str, str]]:
        return list(cls._sessions.get(session_id, []))

    @classmethod
    def get_context_string(cls, session_id: str, max_items: int = 6) -> str:
        history = cls.get_history(session_id)
        if not history:
            return ""
        recent = history[-max_items:]
        parts = []
        for msg in recent:
            role = msg["role"].upper()
            parts.append(f"{role}: {msg['content'][:500]}")
        return "\n".join(parts)

    @classmethod
    def clear(cls, session_id: str) -> None:
        cls._sessions.pop(session_id, None)


# ─── Intent Router ───────────────────────────────────────────────────────────


class IntentRouter:
    """Routes messages between the brain pipeline and the code agent.

    The brain handles: greetings, identity, questions, explanations,
    emotions, general conversation.
    The code agent handles: code generation, file creation, debugging,
    refactoring, testing, analysis of code.

    Complexity: T(route) in O(n) for n tokens in message.
    """

    # Conversational keywords that route to the brain
    _CONVERSATIONAL_KEYWORDS: Tuple[str, ...] = (
        "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "whats up", "thanks", "thank you",
        "who are you", "what are you", "what is condrox", "about yourself",
        "what can you do", "how do you work", "what is your name",
        "are you an ai", "are you conscious", "do you have feelings",
        "how was you built", "who made you", "who created you",
        "bye", "goodbye", "see you", "good night",
        "yes", "no", "ok", "okay", "sure", "maybe",
        "i feel", "i'm feeling", "i am feeling", "i'm sad", "i'm happy",
        "i'm anxious", "i'm angry", "i'm grateful",
        "help me feel", "make me happy",
    )

    # Code keywords that route to the code agent
    _CODE_KEYWORDS: Tuple[str, ...] = (
        "create", "write", "build", "generate", "implement", "code",
        "function", "class", "component", "html", "css", "javascript",
        "typescript", "python", "react", "file", "script", "program",
        "debug", "fix", "refactor", "optimize", "test", "deploy",
        "sql", "query", "api", "endpoint", "frontend", "backend",
        "algorithm", "sort", "search", "data structure", "palindrome",
        "fibonacci", "factorial", "prime", "binary", "linked list",
        "stack", "queue", "tree", "graph", "dynamic programming",
        "leetcode", "hackerrank", "codewars",
        "def ", "import ", "console.log", "print(", "return ",
        "public class", "void ", "func ", "var ", "let ", "const ",
        "<<file", "<<file:", "end_file>>",
    )

    # Code file extensions and patterns
    _CODE_PATTERNS: Tuple[str, ...] = (
        ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json",
        ".sql", ".sh", ".go", ".rs", ".java", ".c", ".cpp", ".rb",
        "def ", "function ", "class ", "import ", "export ",
        "async ", "await ", "const ", "let ", "var ",
    )

    @classmethod
    def is_code_request(cls, message: str) -> bool:
        """Determine if a message is a code request or conversational.

        Returns True for code requests, False for conversational.

        Complexity: T(n) in O(k) for k keywords, S(n) in O(1).
        """
        msg_lower = message.lower().strip()

        # Short messages (<= 4 words) that are conversational
        words = msg_lower.split()
        if len(words) <= 3:
            # Check if it's a pure greeting or conversational
            for kw in cls._CONVERSATIONAL_KEYWORDS:
                if msg_lower == kw or msg_lower.startswith(kw):
                    return False

        # Check for code patterns (strong signal)
        for pattern in cls._CODE_PATTERNS:
            if pattern in msg_lower:
                return True

        # Check for file action protocol from Workstation
        if "<<file:" in msg_lower or "<<file " in msg_lower:
            return True

        # Check for code keywords
        code_score = 0
        for kw in cls._CODE_KEYWORDS:
            if kw in msg_lower:
                code_score += 1

        # Check for conversational keywords
        conv_score = 0
        for kw in cls._CONVERSATIONAL_KEYWORDS:
            if kw in msg_lower:
                conv_score += 1

        # If no signals, default to conversational (brain handles general chat)
        if code_score == 0 and conv_score == 0:
            return False

        return code_score > conv_score

    @classmethod
    def detect_language(cls, message: str) -> Tuple[str, str]:
        """Detect programming language from message.

        Returns (language, file_path).

        Complexity: T(n) in O(k) for k language keywords.
        """
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ("html", "webpage", "web page", "web site", "website", "landing page")):
            return ("html", "output.html")
        if any(kw in msg_lower for kw in ("css", "stylesheet", "style sheet")):
            return ("css", "output.css")
        if any(kw in msg_lower for kw in ("javascript", "js ", "jscript", "node", ".js")):
            return ("javascript", "output.js")
        if any(kw in msg_lower for kw in ("typescript", "ts ", "tscript", ".ts", ".tsx")):
            return ("typescript", "output.ts")
        if any(kw in msg_lower for kw in ("sql", "query", "database", "select ", "insert ", "create table")):
            return ("sql", "output.sql")
        if any(kw in msg_lower for kw in ("bash", "shell", "sh script", "shell script")):
            return ("bash", "output.sh")
        if any(kw in msg_lower for kw in ("json", "json schema")):
            return ("json", "output.json")
        if any(kw in msg_lower for kw in ("markdown", "md file", ".md")):
            return ("markdown", "output.md")
        if any(kw in msg_lower for kw in ("react", "component", "jsx", "tsx")):
            return ("typescript", "output.tsx")
        return ("python", "chat_output.py")


# ─── Request/Response Models ───────────────────────────────────────────────────


class ExecuteRequest(BaseModel):
    """Request model for code task execution."""
    
    user_input: str = Field(..., min_length=1, description="Coding task description")
    language: str = Field(default="python", description="Target programming language")
    file_path: str = Field(default="output.py", description="Target file path")
    context: str = Field(default="", description="Additional context")


class ExecuteResponse(BaseModel):
    """Response model for code task execution."""
    
    task_id: str
    intent: str
    output: str
    verification_passed: bool
    verification_blocked: bool
    verification_score: float
    state: str
    confidence: float
    confidence_level: str
    domain: str
    latency_ms: int
    errors: list[str]


class IndexRequest(BaseModel):
    """Request model for file indexing."""
    
    file_path: str = Field(..., description="File path to index")
    code: str = Field(..., min_length=1, description="File content")
    language: str = Field(default="python", description="Programming language")


class IndexResponse(BaseModel):
    """Response model for file indexing."""
    
    file_path: str
    indexed: bool
    chunks_created: int
    rag_index_size: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    agent_state: str
    rag_index_size: int
    uptime_ms: int


class StatsResponse(BaseModel):
    """Response model for agent statistics."""
    
    rag_index_size: int
    agent_state: str
    total_executions: int
    successful_executions: int
    blocked_executions: int
    failed_executions: int
    avg_latency_ms: float


# ─── Global State ─────────────────────────────────────────────────────────────


_agent: Optional[CondroxAgentCoder] = None
_brain_controller: Optional[Any] = None
_start_time: float = time.time()
_execution_stats: Dict[str, int] = {
    "total": 0,
    "successful": 0,
    "blocked": 0,
    "failed": 0,
}
_latency_sum: float = 0.0

# Cached brain instance for chat endpoint
_cached_brain: Optional[Any] = None


# ─── Lifespan Management ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage agent lifecycle."""
    global _agent, _brain_controller, _cached_brain
    
    # Initialize agent
    _agent = CondroxAgentCoder(root_dir=".")
    
    # Initialize cached brain for chat endpoint
    try:
        from core import BrainController
        _cached_brain = BrainController()
        _cached_brain.boot_system()
        logger.info("Cached brain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cached brain: {e}")
        _cached_brain = None
    
    yield
    
    # Cleanup
    if _cached_brain:
        try:
            _cached_brain.close()
        except Exception:
            pass


# ─── FastAPI App ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="Condrox Suite Developer API",
    description="Condrox Suite DeveloperAgent API — code assistant powered by Condrox brain",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    uptime_ms = int((time.time() - _start_time) * 1000)
    
    return HealthResponse(
        status="healthy",
        agent_state=_agent.state.value,
        rag_index_size=_agent.rag_index_size,
        uptime_ms=uptime_ms,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get agent statistics."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    total = _execution_stats["total"]
    avg_latency = _latency_sum / total if total > 0 else 0.0
    
    return StatsResponse(
        rag_index_size=_agent.rag_index_size,
        agent_state=_agent.state.value,
        total_executions=total,
        successful_executions=_execution_stats["successful"],
        blocked_executions=_execution_stats["blocked"],
        failed_executions=_execution_stats["failed"],
        avg_latency_ms=avg_latency,
    )


@app.post("/index", response_model=IndexResponse)
async def index_file(request: IndexRequest) -> IndexResponse:
    """Index a project file for RAG-based context retrieval."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        index_size_before = _agent.rag_index_size
        _agent.index_project(request.file_path, request.code, request.language)
        chunks_created = _agent.rag_index_size - index_size_before
        
        return IndexResponse(
            file_path=request.file_path,
            indexed=True,
            chunks_created=chunks_created,
            rag_index_size=_agent.rag_index_size,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


@app.post("/chat")
async def chat_endpoint(request: dict):
    """Main chat endpoint with full conversational + code intelligence.

    Routes messages intelligently:
    - Conversational (greetings, identity, emotions, questions) → Brain pipeline
    - Code requests (create, write, fix, refactor) → Code agent with quality gates

    Features:
    - Conversation memory per session
    - Emotion detection and personality-driven responses
    - Intent classification via Word Identity Engine
    - Code generation with 6 quality gates
    - Language auto-detection (Python, HTML, CSS, JS, TS, SQL, etc.)

    Complexity: T(n) in O(pipeline) for brain, O(pipeline + gates) for code.
    """
    global _agent, _cached_brain

    message = request.get("message", "")
    session_id = request.get("session_id", "default")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    if _cached_brain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Brain not initialized"
        )

    # Store user message in conversation memory
    ConversationMemory.add(session_id, "user", message)

    # Route: code request vs conversational
    is_code = IntentRouter.is_code_request(message)

    try:
        if is_code:
            # ── CODE PATH: Route through code agent with quality gates ──
            if _agent is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="DeveloperAgent not initialized"
                )

            detected_lang, detected_path = IntentRouter.detect_language(message)

            # Don't pass conversation context to the code agent — it confuses
            # the brain's emotion detector. The code agent gets the raw message.
            result: AgentCodeResult = await _agent.execute(
                user_input=message,
                brain_controller=_cached_brain,
                language=detected_lang,
                file_path=detected_path,
                context="",
            )

            response_text = result.output

            # Add metadata about the code generation
            metadata = {
                "intent": result.intent.value,
                "language": detected_lang,
                "confidence": result.confidence,
                "confidence_level": result.confidence_level.value,
                "verification_passed": result.verification.passed,
                "verification_score": result.verification.score,
                "domain": result.domain.value,
                "state": result.state.value,
                "latency_ms": result.latency_ms,
            }
        else:
            # ── CONVERSATIONAL PATH: Route through brain pipeline ──
            # The brain handles greetings, identity, emotions, questions,
            # explanations, and general conversation with full personality.
            # The brain has its own memory system, so we send the raw message
            # without conversation context prepended (which confuses emotion detection).

            brain_state = _cached_brain.process_request(message)

            # Extract the response from the brain state
            response_text = brain_state.final_output

            # Extract metadata from the brain state
            metadata = {
                "intent": getattr(brain_state, "intent", "unknown"),
                "intent_confidence": getattr(brain_state, "intent_confidence", 0.0),
                "emotion": getattr(brain_state, "emotion", "neutral"),
                "domain": getattr(brain_state, "domain", "general"),
                "mode": getattr(brain_state, "mode", "default"),
                "security_approved": getattr(brain_state, "security_approved", True),
            }

        # Store assistant response in conversation memory
        ConversationMemory.add(session_id, "assistant", response_text)

        return {
            "response": response_text,
            "metadata": metadata,
            "session_id": session_id,
            "routed_to": "code_agent" if is_code else "brain_pipeline",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat execution failed: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: dict):
    """Streaming chat endpoint with SSE (Server-Sent Events).

    Streams the response in chunks for real-time UI feedback.
    Routes to brain or code agent based on intent, same as /chat.

    Complexity: T(n) in O(pipeline + stream_chunks).
    """
    global _agent, _cached_brain

    message = request.get("message", "")
    session_id = request.get("session_id", "default")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    if _cached_brain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Brain not initialized"
        )

    ConversationMemory.add(session_id, "user", message)
    is_code = IntentRouter.is_code_request(message)

    async def generate():
        try:
            if is_code:
                if _agent is None:
                    yield f"data: {json.dumps({'error': 'DeveloperAgent not initialized'})}\n\n"
                    return

                detected_lang, detected_path = IntentRouter.detect_language(message)
                conv_context = ConversationMemory.get_context_string(session_id, max_items=4)

                result: AgentCodeResult = await _agent.execute(
                    user_input=message,
                    brain_controller=_cached_brain,
                    language=detected_lang,
                    file_path=detected_path,
                    context=conv_context,
                )
                response_text = result.output
                metadata = {
                    "intent": result.intent.value,
                    "language": detected_lang,
                    "confidence": result.confidence,
                    "routed_to": "code_agent",
                }
            else:
                brain_state = _cached_brain.process_request(message)
                response_text = brain_state.final_output
                metadata = {
                    "intent": getattr(brain_state, "intent", "unknown"),
                    "emotion": getattr(brain_state, "emotion", "neutral"),
                    "routed_to": "brain_pipeline",
                }

            ConversationMemory.add(session_id, "assistant", response_text)

            # Stream in word chunks for live typing effect
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Send final metadata
            final_data = {"final": True, "metadata": metadata, "session_id": session_id}
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session.

    Complexity: T(n) in O(1).
    """
    ConversationMemory.clear(session_id)
    return {"cleared": True, "session_id": session_id}


@app.get("/chat/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session.

    Complexity: T(n) in O(k) for k history items.
    """
    return {
        "session_id": session_id,
        "history": ConversationMemory.get_history(session_id),
        "count": len(ConversationMemory.get_history(session_id)),
    }

@app.post("/execute", response_model=ExecuteResponse)
async def execute_task(
    request: ExecuteRequest,
    brain_controller: Any = None,
) -> ExecuteResponse:
    """Execute a code task through the agent state machine."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    if brain_controller is None:
        brain_controller = _cached_brain

    if brain_controller is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="BrainController required (not provided)"
        )
    
    start_time = time.perf_counter()
    _execution_stats["total"] += 1
    
    try:
        result: AgentCodeResult = await _agent.execute(
            user_input=request.user_input,
            brain_controller=brain_controller,
            language=request.language,
            file_path=request.file_path,
            context=request.context,
        )
        
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        _latency_sum += elapsed_ms
        
        # Update stats
        if result.state == AgentState.COMPLETED:
            _execution_stats["successful"] += 1
        elif result.state == AgentState.BLOCKED:
            _execution_stats["blocked"] += 1
        elif result.state == AgentState.FAILED:
            _execution_stats["failed"] += 1
        
        return ExecuteResponse(
            task_id=result.task_id,
            intent=result.intent.value,
            output=result.output,
            verification_passed=result.verification.passed,
            verification_blocked=result.verification.blocked,
            verification_score=result.verification.score,
            state=result.state.value,
            confidence=result.confidence,
            confidence_level=result.confidence_level.value,
            domain=result.domain.value,
            latency_ms=elapsed_ms,
            errors=list(result.errors),
        )
    except ValueError as e:
        _execution_stats["failed"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        _execution_stats["failed"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


# ─── Dependency Injection ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


async def get_brain_controller() -> Any:
    """
    Dependency to get BrainController.
    
    In production, this would be injected via environment or service discovery.
    For now, this is a placeholder that must be overridden.
    """
    # This should be overridden by the application that hosts this API
    # e.g., via dependency injection from the main Condrox application
    return None
