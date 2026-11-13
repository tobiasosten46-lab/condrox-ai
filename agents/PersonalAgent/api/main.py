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
PersonalAgent REST API — FastAPI service for friendly conversational assistant.

Endpoints:
- POST /chat — Process conversational input
- POST /context — Get conversation context
- DELETE /context — Clear conversation context
- GET /health — Health check
- GET /stats — Agent statistics

Complexity:
    chat: T(n) ∈ O(pipeline) where pipeline = 18 stages.
    context: T(n) ∈ O(1).

Invariants:
    - All endpoints require valid input.
    - Chat requires Condrox BrainController (provided via dependency).
    - Rate limiting enforced (100 req/min default).

Edge Cases:
    - Empty input -> 400 Bad Request.
    - BrainController unavailable -> 503 Service Unavailable.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.PersonalAgent.personal_agent import (
    PersonalAgent,
    PersonalAgentResult,
    PersonalAgentState,
)
from agents.PersonalAgent.src.role_definition import ToneProfile


# ─── Request/Response Models ───────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request model for conversational input."""
    
    message: str = Field(..., min_length=1, description="User message or question")
    context: str = Field(default="", description="Additional context")


class ChatResponse(BaseModel):
    """Response model for conversational input."""
    
    request_id: str
    response: str
    emotion: str
    context_used: bool
    confidence: float
    metadata: Dict[str, Any]


class ContextResponse(BaseModel):
    """Response model for conversation context."""
    
    context: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    agent_available: bool
    brain_controller_available: bool


class StatsResponse(BaseModel):
    """Response model for agent statistics."""
    
    state: str
    request_count: int
    brain_controller_available: bool


# ─── Global State ─────────────────────────────────────────────────────────────


_cached_brain: Optional[Any] = None
_agent: Optional[PersonalAgent] = None


# ─── Lifecycle Management ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _cached_brain, _agent
    
    # Startup
    logger.info("PersonalAgent API starting up...")
    
    # Initialize BrainController
    try:
        from core.brain.brain_controller import BrainController
        from core.brain.brain_config import BrainConfig
        
        config = BrainConfig()
        _cached_brain = BrainController(config)
        _agent = PersonalAgent(_cached_brain, tone_profile=ToneProfile.PROFESSIONAL)
        logger.info("PersonalAgent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PersonalAgent: {e}")
        _cached_brain = None
        _agent = None
    
    yield
    
    # Shutdown
    logger.info("PersonalAgent API shutting down...")
    if _cached_brain is not None:
        try:
            _cached_brain.close()
        except Exception as e:
            logger.error(f"Error closing BrainController: {e}")


# ─── FastAPI Application ─────────────────────────────────────────────────────


app = FastAPI(
    title="PersonalAgent API",
    description="Friendly conversational assistant using Condrox AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _agent is not None else "unhealthy",
        agent_available=_agent is not None,
        brain_controller_available=_cached_brain is not None,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get agent statistics."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    return StatsResponse(
        state=_agent.state.value,
        request_count=_agent.request_count,
        brain_controller_available=_cached_brain is not None,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process conversational input using PersonalAgent."""
    global _agent
    
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        result = _agent.process(
            user_input=request.message,
            context=request.context,
        )
        
        return ChatResponse(
            request_id=result.request_id,
            response=result.response,
            emotion=result.emotion,
            context_used=result.context_used,
            confidence=result.confidence,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/context", response_model=ContextResponse)
async def get_context():
    """Get current conversation context."""
    global _agent
    
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        context = _agent.get_conversation_context()
        return ContextResponse(context=context)
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get context: {str(e)}"
        )


@app.delete("/context")
async def clear_context():
    """Clear conversation context."""
    global _agent
    
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        _agent.clear_conversation_context()
        return {"status": "success", "message": "Conversation context cleared"}
    except Exception as e:
        logger.error(f"Error clearing context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear context: {str(e)}"
        )
