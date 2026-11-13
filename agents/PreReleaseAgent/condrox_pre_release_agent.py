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

"""Condrox AI Pre-Release Agent — isolated public chat demo.

Exposes the full 18-stage cognitive pipeline to the public website with
zero cross-user contamination. No accounts, no persistent memory, no
shared state between sessions.

Architecture:
- Dedicated BrainController (separate from main API's brain)
- store_memory=False on every process_request call
- Per-session ephemeral context dict (in-memory, TTL'd)
- Conversation history capped at MAX_HISTORY_TURNS
- Input/output length capped for demo snappiness

Complexity:
    chat: T(n) ∈ O(pipeline) ≈ O(Σ s_i), S(n) ∈ O(history + output)
    cleanup_expired: T(n) ∈ O(s) for s sessions, S(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Conversational Engine ───────────────────────────────────────────────────
from core.conversational.conversational_engine import ConversationalEngine

# ─── Unified Cognitive Pipeline ──────────────────────────────────────────────
# Én pipeline som binder sammen cognitive brain, autonomous learning, og
# free will response generation i én sammenhengende data flow.
from core.cognitive_brain import UnifiedCognitivePipeline

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_INPUT_CHARS: int = 4000
MAX_OUTPUT_CHARS: int = 4000
MAX_HISTORY_TURNS: int = 20
SESSION_TTL_SECONDS: int = 1800  # 30 minutes idle
CLEANUP_INTERVAL_SECONDS: int = 300  # 5 minutes

# Path to the self-knowledge JSONL file (next to this module)
_SELF_KNOWLEDGE_PATH: Path = Path(__file__).parent / "condrox_self_knowledge.jsonl"
_BASIC_KNOWLEDGE_PACK_PATH: Path = Path(__file__).parent / "basic_knowledge_pack.jsonl"


@dataclass(frozen=True, slots=True)
class PreReleaseResult:
    """Result of a pre-release chat request.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(output length).
    """

    session_id: str
    response: str
    intent: str
    domain: str
    mode: str
    confidence: float
    truncated: bool
    errors: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    follow_up_question: str = ""
    topics: List[str] = field(default_factory=list)
    curiosity_score: float = 0.0


@dataclass(slots=True)
class PreReleaseSession:
    """Ephemeral per-session conversation state.

    Lives only in process memory. Never persisted to disk.
    Cleared after SESSION_TTL_SECONDS of inactivity.

    Complexity: T(n) ∈ O(1) per operation, S(n) ∈ O(history).
    """

    session_id: str
    history: List[Tuple[str, str]] = field(default_factory=list)  # (user, ai) turns
    last_active: float = field(default_factory=time.time)
    turn_count: int = 0

    def add_turn(self, user_msg: str, ai_msg: str) -> None:
        """Add a conversation turn, trimming to MAX_HISTORY_TURNS."""
        self.history.append((user_msg, ai_msg))
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history = self.history[-MAX_HISTORY_TURNS:]
        self.turn_count += 1
        self.last_active = time.time()

    def is_expired(self, now: float) -> bool:
        """Check if session has been idle longer than TTL."""
        return (now - self.last_active) > SESSION_TTL_SECONDS

    def build_context(self) -> Dict[str, Any]:
        """Build context dict from conversation history for the pipeline.

        Complexity: T(n) ∈ O(h) for h history turns, S(n) ∈ O(h).
        """
        if not self.history:
            return {}
        # Provide the last few turns as conversational context
        # so the pipeline can do multi-turn recall within the session
        context: Dict[str, Any] = {}
        history_text = []
        for user_msg, ai_msg in self.history[-MAX_HISTORY_TURNS:]:
            history_text.append(f"User: {user_msg}")
            history_text.append(f"Assistant: {ai_msg}")
        context["conversation_history"] = "\n".join(history_text)
        context["session_id"] = self.session_id
        context["pre_release"] = True
        return context


class PreReleaseAgent:
    """Public, isolated, no-auth chat agent for the Condrox AI demo.

    Wraps a dedicated BrainController instance with strict isolation:
    - No persistent memory writes (store_memory=False)
    - Per-session ephemeral context
    - Length caps on input and output
    - Periodic cleanup of expired sessions

    Thread-safety: A reentrant lock guards the sessions dict. The
    BrainController itself has a single-instance lock for pipeline runs.

    Complexity:
        chat:       T(n) ∈ O(pipeline), S(n) ∈ O(history + output)
        get_session: T(n) ∈ O(1)
        cleanup:    T(n) ∈ O(s) for s sessions
    """

    __slots__ = ("_brain", "_sessions", "_lock", "_last_cleanup", "_booted", "_conv_engine", "_unified")

    def __init__(self, brain_controller: Any) -> None:
        """Initialize with a dedicated BrainController instance.

        Args:
            brain_controller: A booted BrainController dedicated to this agent.
                Must NOT be shared with the main API to guarantee isolation.
        """
        self._brain: Any = brain_controller
        self._sessions: Dict[str, PreReleaseSession] = {}
        self._lock: threading.RLock = threading.RLock()
        self._last_cleanup: float = time.time()
        self._booted: bool = False
        self._conv_engine: ConversationalEngine = ConversationalEngine()
        self._unified: UnifiedCognitivePipeline = UnifiedCognitivePipeline()

    def boot(self) -> None:
        """Boot the underlying BrainController if not already booted.

        All task-specific agents (Code, Analyze, Architect, Explain) remain
        registered so the PreReleaseAgent can handle real programming,
        analysis, architecture, and explanation requests. The DefaultAgent
        handles general conversation, greetings, identity, and emotional
        support.
        """
        if not self._booted:
            if self._brain._boot_report is None:
                logger.info("PreReleaseAgent: booting BrainController...")
                self._brain.boot_system()
                logger.info("PreReleaseAgent: BrainController booted.")
            # All agents remain registered — full AI capabilities enabled.
            # The pipeline routes to the correct agent based on intent:
            #   code → CodeAgent, analyze → AnalyzeAgent,
            #   architect → ArchitectAgent, explain → ExplainAgent,
            #   everything else → DefaultAgent
            try:
                agent_count = self._brain.agent_registry.count()
                agent_modes = list(self._brain.agent_registry.modes())
                logger.info(
                    "PreReleaseAgent: %d agents registered: %s",
                    agent_count, agent_modes,
                )
            except Exception:
                pass
            self._booted = True

    def _ingest_self_knowledge(self) -> int:
        """Ingest Condrox AI self-knowledge into the brain's knowledge backend.

        Reads the condrox_self_knowledge.jsonl file and ingests each chunk
        into the knowledge backend with high confidence (1.0) in the "general"
        domain. This ensures that queries about Condrox AI itself retrieve
        accurate, relevant knowledge instead of random dataset content.

        Must be called after boot(). Safe to call multiple times — uses
        item_id from the JSONL to deduplicate.

        Returns:
            Number of knowledge items ingested.

        Complexity: T(n) ∈ O(n * c) for n chunks, c avg chunk length.
        """
        if not self._booted:
            self.boot()

        if not _SELF_KNOWLEDGE_PATH.exists():
            logger.warning(
                "PreReleaseAgent: self-knowledge file not found at %s",
                _SELF_KNOWLEDGE_PATH,
            )
            return 0

        backend = self._brain.knowledge_router.backend
        if backend is None:
            logger.warning("PreReleaseAgent: knowledge backend is None, skipping self-knowledge ingestion.")
            return 0

        ingested: int = 0
        try:
            with open(_SELF_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        item_id = obj.get("id", "")
                        if len(text) < 20 or not item_id:
                            continue
                        # Check if already ingested (dedup by item_id)
                        if backend.get(item_id) is not None:
                            continue
                        self._brain.knowledge_router.ingest(
                            content=text,
                            domain="general",
                            item_id=item_id,
                            confidence=1.0,
                            metadata={
                                "source": "condrox_self_knowledge",
                                "self": True,
                            },
                        )
                        ingested += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as exc:
            logger.error("PreReleaseAgent: failed to ingest self-knowledge: %s", exc)
            return ingested

        # Rebuild the knowledge graph so the new items are connected
        if ingested > 0:
            try:
                kb_count = backend.count()
                if 0 < kb_count <= 5000:
                    self._brain.knowledge_router.build_graph()
            except Exception:
                pass  # Non-fatal — graph is optional
            logger.info(
                "PreReleaseAgent: ingested %d self-knowledge items.", ingested
            )
        return ingested

    def _ingest_basic_knowledge_pack(self) -> int:
        """Ingest the basic knowledge pack into the brain's knowledge backend.

        Reads the basic_knowledge_pack.jsonl file and ingests each chunk
        into the knowledge backend with the domain specified in the JSONL.
        This provides the system with enough basic knowledge to answer
        common questions across 12 domains without falling back to
        irrelevant dataset content.

        Must be called after boot(). Safe to call multiple times — uses
        item_id from the JSONL to deduplicate.

        Returns:
            Number of knowledge items ingested.

        Complexity: T(n) ∈ O(n * c) for n chunks, c avg chunk length.
        """
        if not self._booted:
            self.boot()

        if not _BASIC_KNOWLEDGE_PACK_PATH.exists():
            logger.warning(
                "PreReleaseAgent: basic knowledge pack not found at %s",
                _BASIC_KNOWLEDGE_PACK_PATH,
            )
            return 0

        backend = self._brain.knowledge_router.backend
        if backend is None:
            logger.warning("PreReleaseAgent: knowledge backend is None, skipping basic knowledge pack.")
            return 0

        ingested: int = 0
        try:
            with open(_BASIC_KNOWLEDGE_PACK_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        item_id = obj.get("id", "")
                        domain = obj.get("domain", "general")
                        if len(text) < 20 or not item_id:
                            continue
                        if backend.get(item_id) is not None:
                            continue
                        self._brain.knowledge_router.ingest(
                            content=text,
                            domain=domain,
                            item_id=item_id,
                            confidence=1.0,
                            metadata={
                                "source": "basic_knowledge_pack",
                                "pack": "v1.0",
                            },
                        )
                        ingested += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as exc:
            logger.error("PreReleaseAgent: failed to ingest basic knowledge pack: %s", exc)
            return ingested

        if ingested > 0:
            try:
                kb_count = backend.count()
                if 0 < kb_count <= 5000:
                    self._brain.knowledge_router.build_graph()
            except Exception:
                pass
            logger.info(
                "PreReleaseAgent: ingested %d basic knowledge pack items.", ingested
            )
        return ingested

    def get_or_create_session(self, session_id: Optional[str] = None) -> PreReleaseSession:
        """Get an existing session or create a new one.

        Args:
            session_id: Optional existing session ID. If None or unknown,
                a new session is created.

        Returns:
            The PreReleaseSession for this client.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._maybe_cleanup()
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            new_id = session_id or f"pre_{uuid.uuid4().hex[:12]}"
            session = PreReleaseSession(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def chat(self, message: str, session_id: Optional[str] = None) -> PreReleaseResult:
        """Process a chat message and return an isolated response.

        Args:
            message: User input text (capped at MAX_INPUT_CHARS).
            session_id: Optional session ID for multi-turn context.

        Returns:
            PreReleaseResult with the response and metadata.

        Raises:
            ValueError: If message is empty.
            RuntimeError: If the brain is not initialized.

        Complexity: T(n) ∈ O(pipeline), S(n) ∈ O(history + output).
        """
        if not message or not message.strip():
            raise ValueError("message must be non-empty")

        if self._brain is None:
            raise RuntimeError("BrainController not initialized")

        # Cap input length
        truncated_input = message[:MAX_INPUT_CHARS]
        was_truncated = len(message) > MAX_INPUT_CHARS

        # Get/create session and build context
        session = self.get_or_create_session(session_id)
        context = session.build_context()

        # Run the pipeline with store_memory=False (zero contamination)
        # The single-instance lock is disabled in BrainConfig for the
        # pre-release brain so concurrent requests don't block.
        state = self._brain.process_request(
            raw_input=truncated_input,
            context=context,
            store_memory=False,
        )

        # ─── UNIFIED COGNITIVE PIPELINE ───────────────────────────────
        # ÉN data flow i korrekt logisk rekkefølge:
        #   1. Vision context (hvem er jeg)
        #   2. Goal check (hva prøver jeg å oppnå)
        #   3. Opinion check (hva tenker jeg om dette)
        #   4. Cognitive phases (reasoning, pattern, analysis, reverse, structure)
        #   5. Knowledge retrieval (formet av goals, opinions, phases)
        #   6. Response generation (free will, informert av ALT)
        #   7. Vision alignment (er responsen aligned)
        #   8. Reward (kognitiv verdi)
        #   9. Experience recording (lagre som kognitiv struktur)
        #   10. Goal progress (førte dette nærmere et mål)
        #   11. Opinion update (endret dette meninger)
        #   12. Meta-learning (forbedret dette strategier)
        #
        # Pipeline output (from 18-stage pipeline agents) is passed in
        # as pipeline_output so the unified pipeline can use it for
        # task intents (code, analyze, architect) while generating
        # its own responses for conversational and question intents.
        conversation_history = [t[0] for t in session.history[-5:]]
        try:
            ctx = self._unified.process(
                user_message=truncated_input,
                intent=state.intent,
                emotion=state.emotion,
                domain=state.domain,
                mode=state.mode,
                confidence=state.intent_confidence,
                conversation_history=conversation_history,
                session_id=session.session_id,
                pipeline_output=state.final_output or "",
            )
            output = ctx.generated_response
        except Exception as exc:
            logger.debug("Unified pipeline failed (non-fatal): %s", exc)
            output = state.final_output or ""

        # Cap output length
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS].rstrip() + "…"
            was_truncated = True

        # ─── Conversational Engine: topic tracking + follow-up questions ───
        topics: List[str] = []
        follow_up: str = ""
        curiosity_score: float = 0.0
        try:
            _topics, _follow_up = self._conv_engine.process_turn(
                session_id=session.session_id,
                user_input=truncated_input,
                intent=state.intent,
                emotion=state.emotion,
            )
            topics = list(_topics) if _topics else []
            follow_up = _follow_up or ""  # Convert None to empty string
            # Enrich response with context-aware personalization
            output = self._conv_engine.enrich_response(
                session_id=session.session_id,
                response=output,
                intent=state.intent,
                emotion=state.emotion,
            )
            # Inject follow-up into the response
            if follow_up:
                output = self._conv_engine.inject_follow_up(output, follow_up)
                # Re-cap if follow-up pushed us over the limit
                if len(output) > MAX_OUTPUT_CHARS:
                    output = output[:MAX_OUTPUT_CHARS].rstrip() + "…"
                    was_truncated = True
            # Get curiosity score for the profile
            profile = self._conv_engine.get_profile(session.session_id)
            curiosity_score = self._conv_engine._curiosity.curiosity_score(profile)
            # Anti-repetition: detect if this response is too similar to a previous one
            if profile.is_repetitive_response(output):
                # Add a variation to break the repetition
                import random as _rnd
                variations = [
                    " Let me put it another way.",
                    " From a different angle:",
                    " Thinking about it differently —",
                ]
                # Insert variation after the first sentence
                first_period = output.find(". ")
                if first_period != -1 and first_period < 200:
                    output = output[:first_period + 1] + _rnd.choice(variations) + output[first_period + 1:]
            # Record this response for future anti-repetition checks
            profile.record_response(output)
        except Exception as exc:
            logger.debug("ConversationalEngine error (non-fatal): %s", exc)

        # Store turn in ephemeral session history
        session.add_turn(truncated_input, output)

        # NOTE: All learning happens INSIDE the unified pipeline above.
        # The pipeline already:
        #   - Extracted knowledge (stage 9)
        #   - Validated and stored facts (stage 9)
        #   - Recorded experience (stage 9)
        #   - Updated goals (stage 10)
        #   - Formed opinions (stage 11)
        #   - Recorded meta-learning (stage 12)
        # No separate learning calls needed — it's all one flow.

        return PreReleaseResult(
            session_id=session.session_id,
            response=output,
            intent=state.intent,
            domain=state.domain,
            mode=state.mode,
            confidence=state.intent_confidence,
            truncated=was_truncated,
            errors=list(state.errors),
            emotion=state.emotion,
            follow_up_question=follow_up,
            topics=topics,
            curiosity_score=curiosity_score,
        )

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session's history.

        Returns:
            True if the session was found and cleared, False otherwise.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._conv_engine.clear_session(session_id)
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def session_count(self) -> int:
        """Return the current number of active sessions.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return len(self._sessions)

    def _maybe_cleanup(self) -> int:
        """Clean up expired sessions if cleanup interval has elapsed.

        Returns:
            Number of sessions removed.

        Complexity: T(n) ∈ O(s) for s sessions.
        """
        now = time.time()
        if (now - self._last_cleanup) < CLEANUP_INTERVAL_SECONDS:
            return 0
        self._last_cleanup = now
        expired_ids = [
            sid for sid, s in self._sessions.items() if s.is_expired(now)
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        if expired_ids:
            logger.info(
                f"PreReleaseAgent: cleaned up {len(expired_ids)} expired sessions."
            )
        return len(expired_ids)

    def health(self) -> Dict[str, Any]:
        """Return health status for the /health endpoint.

        Complexity: T(n) ∈ O(1).
        """
        brain_ok = self._brain is not None and self._booted
        agent_count = 0
        agent_modes: List[str] = []
        if brain_ok:
            try:
                agent_count = self._brain.agent_registry.count()
                agent_modes = list(self._brain.agent_registry.modes())
            except Exception:
                pass

        # Unified cognitive pipeline stats — ONE system, not separate
        unified_stats: Dict[str, Any] = {}
        try:
            ustate = self._unified.get_state()
            ustats = self._unified.get_stats()
            unified_stats = {
                "total_cycles": ustate["total_cycles"],
                "total_reward": ustate["total_reward"],
                "breakthroughs": ustate["breakthroughs"],
                "stable_facts": ustate["stable_facts"],
                "active_goals": ustate["active_goals"],
                "alignment_trend": ustate["alignment_trend"],
                "total_opinions": ustate["total_opinions"],
                "total_experiences": ustate["total_experiences"],
                "knowledge_facts": ustate["knowledge_facts"],
                "knowledge_concepts": ustate["knowledge_concepts"],
                "patterns": ustats["patterns"],
                "meta_outcomes": ustats["meta"]["outcomes"],
                "meta_domains": ustats["meta"]["domains"],
                "meta_strategies": ustats["meta"]["strategies"],
            }
        except Exception:
            pass

        return {
            "status": "healthy" if brain_ok else "booting",
            "agent": "PreReleaseAgent",
            "brain_booted": brain_ok,
            "active_sessions": self.session_count(),
            "max_history_turns": MAX_HISTORY_TURNS,
            "max_input_chars": MAX_INPUT_CHARS,
            "max_output_chars": MAX_OUTPUT_CHARS,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "agent_count": agent_count,
            "agent_modes": agent_modes,
            "unified_pipeline": unified_stats,
        }
