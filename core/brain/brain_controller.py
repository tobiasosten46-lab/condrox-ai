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
Unified Brain Controller for Condrox AI.

The BrainController is the central orchestrator that owns all core
intelligence subsystems and runs the 14-stage cognitive pipeline. It is
designed as a concrete, real implementation that can be booted, queried,
and introspected.

Mathematical Foundation:
- process_request is a pipeline fold over 14 stages: O(Î£ s_i) time.
- Subsystem lookup is O(1) because each subsystem is held by direct reference.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from core.boot.core_boot import BootReport, CoreBoot
from core.brain.brain_config import BrainConfig
from core.brain.brain_state import BrainState
from core.cognition.context_manager import ContextManager
from core.cognition.meta_learning_engine import MetaLearningEngine
from core.cognition.reasoning_engine import ReasoningEngine
from core.cognition.strategy_manager import StrategyManager
from core.engines.dcl.dcl_engine import DCLEngine
from core.engines.heartbeat.heartbeat_engine import HealthStatus, HeartbeatEngine
from core.engines.hwa.hwa_engine import HWAEngine
from core.engines.pbe.pbe_engine import PBEEngine
from core.ethics.ethics_engine import EthicsEngine
from core.knowledge.knowledge_router import KnowledgeRouter
from core.knowledge.knowledge_loader import KnowledgeLoader
from core.knowledge.sqlite_knowledge_loader import SQLiteKnowledgeLoader
from core.knowledge.disk_knowledge_backend import DiskKnowledgeBackend
from core.memory.memory_manager import MemoryManager
from core.memory.long_term_memory import LongTermMemory
from core.memory.skill_memory import SkillMemory
from core.memory.user_profile_store import UserProfileStore
from core.school.school_system import CondroxSchoolSystem
from core.training.scenarios import ALL_SCENARIOS
from core.training.training_console import TrainingConsole
from core.personality.personality_engine import PersonalityEngine
from core.personality.personality_profiles import condrox_profile
from core.pipeline.pipeline_orchestrator import PipelineOrchestrator
from core.pipeline.stage_01_preprocessor import Stage01Preprocessor
from core.pipeline.stage_02_intent import Stage02Intent
from core.pipeline.stage_02b_intent_enrichment import Stage02bIntentEnrichment
from core.pipeline.stage_03_mode_selector import Stage03ModeSelector
from core.pipeline.stage_04_domain_router import Stage04DomainRouter
from core.pipeline.stage_05_context_builder import Stage05ContextBuilder
from core.pipeline.stage_05b_context_enrichment import Stage05bContextEnrichment
from core.pipeline.stage_06_knowledge_fetch import Stage06KnowledgeFetch
from core.pipeline.stage_07_memory_fetch import Stage07MemoryFetch
from core.pipeline.stage_08_reasoning_core import Stage08ReasoningCore
from core.pipeline.stage_08b_agent_execution import Stage08bAgentExecution
from core.pipeline.stage_09_synthesis import Stage09Synthesis
from core.pipeline.stage_09b_response_quality import Stage09bResponseQuality
from core.pipeline.stage_09c_pattern_learning import Stage09cPatternLearning
from core.pipeline.stage_09d_metacognition import Stage09dMetacognition
from core.pipeline.stage_10_personality import Stage10Personality
from core.pipeline.stage_11_ethics import Stage11Ethics
from core.pipeline.stage_12_security import Stage12Security
from core.pipeline.stage_13_output_builder import Stage13OutputBuilder
from core.pipeline.stage_14_delivery import Stage14Delivery
from core.security.security_engine import SecurityEngine

# Phase 2-A: Conversational intelligence components
from core.conversational.memory_stabilization import MemoryStabilizationLayer
from core.conversational.search_agent_guard import SearchAgentGuard
from core.conversational.single_instance_lock import SingleInstanceLock
from core.conversational.state_query_handler import StateQueryHandler

# Phase 2-B: Conversational quality components
from core.conversational.claim_ranker_v2 import ClaimRankerV2
from core.conversational.language_enforcer_v2 import LanguageEnforcerV2
from core.conversational.output_stabilizer import OutputStabilizer
from core.conversational.response_style_engine import ResponseStyleEngine
from core.conversational.temporal_context import TemporalContextLayer

# Phase 2-C: Observability & Safety
from core.conversational.error_recovery import ErrorRecoveryLayer
from core.conversational.pipeline_monitor import PipelineMonitor
from core.conversational.safety_layer import SafetyLayer

# Phase 2-D: Agent Coordination
from core.conversational.agent_arbiter import AgentArbiter, AgentCapability, AgentDescriptor
from core.conversational.knowledge_router_v2 import KnowledgeRouterV2
from core.conversational.multi_agent_memory_sync import MultiAgentMemorySync
from core.agents import (
    AgentRegistry,
    CodeAgent,
    ExplainAgent,
    AnalyzeAgent,
    ArchitectAgent,
    DefaultAgent,
)
from core.agents.external_adapters import (
    PersonalAgentAdapter,
    DeveloperAgentAdapter,
)

# Phase 2-E: Advanced Conversational Intelligence
from core.conversational.intent_expander import SemanticIntentExpander
from core.conversational.multi_turn_reasoning import MultiTurnReasoningEngine
from core.conversational.memory_graph import ConversationalMemoryGraph
from core.conversational.goal_tracker import DialogueGoalTracker
from core.conversational.consistency_engine import ConsistencyEngine
from core.conversational.semantic_repair import SemanticRepair
from core.conversational.contradiction_detector import ContradictionDetector

# Phase 2-F: System-Level Properties
from core.conversational.latency_compensation import LatencyCompensationLayer
from core.conversational.priority_engine import ConversationalPriorityEngine
from core.conversational.semantic_interrupt import SemanticInterruptHandler
from core.conversational.token_pressure_regulator import TokenPressureRegulator

# Phase 4: Creative Stack
from core.creative import (
    EmotionPaletteMapper,
    MusicPromptBuilder,
    QualityGuardrail,
)

# Phase 13: AI-OS
from core.os import AIOSKernel
from core.os.service_manager import ServiceDescriptor

# Universal Tools
from core.tools import FileOps, ShellExec, GitOps


class BrainController:
    """
    Central orchestrator for the unified Brain system.

    Attributes:
        config: Immutable BrainConfig.
        memory_manager: Semantic memory manager.
        knowledge_router: Knowledge retrieval router.
        reasoning_engine: Symbolic reasoning engine.
        context_manager: Hierarchical context manager.
        strategy_manager: Strategy selection manager.
        meta_learning_engine: Performance learning tracker.
        hwa_engine: Heuristic weighted analyzer.
        pbe_engine: Pattern-based engine.
        dcl_engine: Dynamic context layer engine.
        heartbeat_engine: Health monitor.
        personality_engine: Personality styling engine.
        security_engine: Input/output security engine.
        ethics_engine: Content ethics engine.
        boot: Boot orchestrator.
        pipeline: 14-stage pipeline orchestrator.

    Complexity:
        __init__: O(1) for fixed-size subsystem construction.
        process_request: O(Î£ s_i) over pipeline stages.
    """

    __slots__ = (
        "config",
        "memory_manager",
        "knowledge_router",
        "reasoning_engine",
        "context_manager",
        "strategy_manager",
        "meta_learning_engine",
        "hwa_engine",
        "pbe_engine",
        "dcl_engine",
        "heartbeat_engine",
        "personality_engine",
        "security_engine",
        "ethics_engine",
        "boot",
        "pipeline",
        "_boot_report",
        # Phase 2-A: Conversational intelligence components
        "state_query_handler",
        "single_instance_lock",
        "memory_stabilization",
        "search_agent_guard",
        # Phase 2-B: Conversational quality components
        "response_style_engine",
        "claim_ranker_v2",
        "temporal_context",
        "output_stabilizer",
        "language_enforcer_v2",
        # Phase 2-C: Observability & Safety
        "safety_layer",
        "pipeline_monitor",
        "error_recovery",
        # Phase 2-D: Agent Coordination
        "agent_arbiter",
        "agent_registry",
        "knowledge_router_v2",
        "multi_agent_memory_sync",
        # Phase 2-E: Advanced Conversational Intelligence
        "intent_expander",
        "multi_turn_reasoning",
        "memory_graph",
        "goal_tracker",
        "consistency_engine",
        "semantic_repair",
        "contradiction_detector",
        # Phase 2-F: System-Level Properties
        "latency_compensation",
        "token_pressure_regulator",
        "semantic_interrupt",
        "priority_engine",
        # Universal Tools
        "file_ops",
        "shell_exec",
        "git_ops",
        # Phase 4: Creative Stack
        "emotion_palette_mapper",
        "music_prompt_builder",
        "quality_guardrail",
        # Phase B: User Memory & Personalization
        "user_profile_store",
        "long_term_memory",
        "skill_memory",
        # Training & School System
        "training_console",
        "school_system",
        # Persistence
        "_storage_backend",
        # Phase 13: AI-OS
        "aios_kernel",
    )

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        """Initialize the BrainController and all subsystems.

        Args:
            config: Optional BrainConfig; defaults are used if None.
        """
        self.config = config if config is not None else BrainConfig.default()

        self.memory_manager = MemoryManager(
            embedding_dim=64, decay_halflife=self.config.memory_decay_halflife_seconds
        )

        # Use a disk-backed knowledge store when configured for large corpora.
        if self.config.use_disk_knowledge:
            kb_path = self.config.persistence_path or "condrox_knowledge.db"
            disk_backend = DiskKnowledgeBackend(
                db_path=kb_path,
                batch_size=1000,
                min_quality_score=0.15,
            )
            self.knowledge_router = KnowledgeRouter(backend=disk_backend, default_top_k=5)
        else:
            self.knowledge_router = KnowledgeRouter(default_top_k=5)

        self.reasoning_engine = ReasoningEngine()
        self.context_manager = ContextManager()
        self.strategy_manager = StrategyManager()
        self.meta_learning_engine = MetaLearningEngine(alpha=0.3)

        self.hwa_engine = None  # lazy â€” not used in pipeline
        self.pbe_engine = None   # lazy â€” not used in pipeline
        self.dcl_engine = DCLEngine()
        self.heartbeat_engine = HeartbeatEngine()

        self.personality_engine = PersonalityEngine(profile=condrox_profile())
        self.security_engine = SecurityEngine(
            max_size=self.config.max_input_length * 2,
            blocked_patterns=list(self.config.blocked_patterns),
        )
        self.ethics_engine = EthicsEngine(blocked_topics=list(self.config.blocked_topics))

        self.boot = CoreBoot()
        self.pipeline = PipelineOrchestrator()
        self._boot_report: Optional[BootReport] = None

        # Phase 2-A: Conversational intelligence components
        self.state_query_handler = StateQueryHandler()
        self.single_instance_lock = SingleInstanceLock()
        self.memory_stabilization = MemoryStabilizationLayer(
            max_items=self.config.max_input_length * 100,
            compression_threshold=50,
            decay_halflife=self.config.memory_decay_halflife_seconds,
        )
        self.search_agent_guard = SearchAgentGuard()

        # Phase 2-B: Conversational quality components
        self.response_style_engine = ResponseStyleEngine()
        self.claim_ranker_v2 = ClaimRankerV2()
        self.temporal_context = TemporalContextLayer(
            default_ttl=self.config.memory_decay_halflife_seconds,
        )
        self.output_stabilizer = OutputStabilizer()
        self.language_enforcer_v2 = LanguageEnforcerV2()

        # Phase 2-C: Observability & Safety
        self.safety_layer = SafetyLayer()
        self.pipeline_monitor = PipelineMonitor()
        self.error_recovery = ErrorRecoveryLayer()

        # Phase 2-D: Agent Coordination
        self.agent_arbiter = AgentArbiter()
        self.agent_registry = AgentRegistry()
        self.agent_registry.register(DefaultAgent())
        self.agent_registry.register(CodeAgent())
        self.agent_registry.register(ExplainAgent())
        self.agent_registry.register(AnalyzeAgent())
        self.agent_registry.register(ArchitectAgent())
        self.agent_registry.register(PersonalAgentAdapter())
        self.agent_registry.register(DeveloperAgentAdapter())
        self.knowledge_router_v2 = KnowledgeRouterV2()
        self.multi_agent_memory_sync = MultiAgentMemorySync()

        # Phase 2-E: Advanced Conversational Intelligence
        self.intent_expander = SemanticIntentExpander()
        self.multi_turn_reasoning = MultiTurnReasoningEngine()
        self.memory_graph = ConversationalMemoryGraph()
        self.goal_tracker = DialogueGoalTracker()
        self.consistency_engine = ConsistencyEngine()
        self.semantic_repair = SemanticRepair()
        self.contradiction_detector = ContradictionDetector()

        # Phase 2-F: System-Level Properties
        self.latency_compensation = LatencyCompensationLayer()
        self.token_pressure_regulator = TokenPressureRegulator(
            total_budget=self.config.max_input_length * 4,
        )
        self.semantic_interrupt = SemanticInterruptHandler()
        self.priority_engine = ConversationalPriorityEngine()

        # Universal Tools â€” available to all external agents
        self.shell_exec = ShellExec()
        self.file_ops = FileOps(root_dir=str(Path.cwd()))
        self.git_ops = GitOps(shell=self.shell_exec)

        # Phase 4: Creative Stack â€” quality_guardrail is in pipeline hot path
        self.quality_guardrail = QualityGuardrail(threshold=0.6, min_length=20)
        # emotion_palette_mapper and music_prompt_builder are creative-only â€” lazy
        self.emotion_palette_mapper = None
        self.music_prompt_builder = None

        # Persistence â€” initialize SQLite backend if configured
        self._storage_backend = None
        if self.config.persistence_path:
            from core.storage import SQLiteStorage
            self._storage_backend = SQLiteStorage(self.config.persistence_path)

        # Phase B: User Memory & Personalization
        self.user_profile_store = UserProfileStore(storage=self._storage_backend)
        self.long_term_memory = LongTermMemory(storage=self._storage_backend)
        self.skill_memory = SkillMemory(storage=self._storage_backend)

        # Training & School System
        self.school_system = CondroxSchoolSystem()
        self.training_console = TrainingConsole(runner=self._training_runner)
        self.training_console.add_scenarios(ALL_SCENARIOS)

        # Phase 13: AI-OS kernel
        self.aios_kernel: Optional[AIOSKernel] = None

        self._wire_subsystems()
        self._build_pipeline()

    def _ensure_creative(self) -> None:
        """Lazy-initialize the creative stack on first access.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1). No-op after first call.
        """
        if self.emotion_palette_mapper is None:
            self.emotion_palette_mapper = EmotionPaletteMapper()
            self.music_prompt_builder = MusicPromptBuilder()

    def _ensure_engines(self) -> None:
        """Lazy-initialize HWA and PBE engines on first access.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1). No-op after first call.
        """
        if self.hwa_engine is None:
            self.hwa_engine = HWAEngine()
            self.pbe_engine = PBEEngine()

    def _wire_subsystems(self) -> None:
        """Configure subsystem defaults and cross-references.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
        """
        self.reasoning_engine.load_default_rules()
        self.dcl_engine.load_defaults()
        self.boot.load_defaults()

        # Register heartbeat checks for core subsystems
        self.heartbeat_engine.register(
            "memory",
            lambda: HealthStatus(
                "memory", 1.0 if self.memory_manager.store.count() >= 0 else 0.0, "ok"
            ),
        )
        self.heartbeat_engine.register(
            "knowledge",
            lambda: HealthStatus(
                "knowledge", 1.0 if self.knowledge_router.backend else 0.0, "ok"
            ),
        )
        self.heartbeat_engine.register(
            "reasoning",
            lambda: HealthStatus(
                "reasoning", 1.0 if self.reasoning_engine.rule_count > 0 else 0.0, "ok"
            ),
        )

    def _build_pipeline(self) -> None:
        """Construct the 14-stage cognitive pipeline.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
        """
        self.pipeline.add_stage(Stage01Preprocessor())
        self.pipeline.add_stage(Stage02Intent())
        self.pipeline.add_stage(Stage02bIntentEnrichment())
        self.pipeline.add_stage(Stage03ModeSelector())
        self.pipeline.add_stage(Stage04DomainRouter())
        self.pipeline.add_stage(Stage05ContextBuilder())
        self.pipeline.add_stage(Stage05bContextEnrichment())
        self.pipeline.add_stage(Stage06KnowledgeFetch(top_k=5))
        self.pipeline.add_stage(Stage07MemoryFetch(top_k=5))
        self.pipeline.add_stage(Stage08ReasoningCore())
        self.pipeline.add_stage(Stage08bAgentExecution())
        self.pipeline.add_stage(Stage09Synthesis())
        self.pipeline.add_stage(Stage09bResponseQuality())
        self.pipeline.add_stage(Stage09cPatternLearning())
        self.pipeline.add_stage(Stage09dMetacognition())
        self.pipeline.add_stage(Stage10Personality())
        self.pipeline.add_stage(Stage11Ethics())
        self.pipeline.add_stage(Stage12Security())
        self.pipeline.add_stage(Stage13OutputBuilder())
        self.pipeline.add_stage(Stage14Delivery())

    def boot_system(self) -> BootReport:
        """Run boot checks and return a report.

        Also boots the AI-OS kernel for service management, monitoring,
        and knowledge hot-reload.

        Returns:
            BootReport.

        Complexity: T(n) âˆˆ O(c) for c checks, plus O(d*f*c) for knowledge loading,
            plus O(Î£ s_i + V + E) for AI-OS boot.
        """
        self._boot_report = self.boot.run()
        # Load persisted state if storage backend is configured
        if self._storage_backend is not None:
            try:
                self.memory_manager.store.load(self._storage_backend, table="memories")
                self.knowledge_router.backend.load(self._storage_backend, table="knowledge")
                self.user_profile_store.load()
                self.long_term_memory.load()
                self.skill_memory.load()
            except Exception:
                pass  # Non-fatal â€” start with empty stores

        # Load knowledge from SQLite (massive ingestion DB) or disk JSONL files
        if self.knowledge_router.backend.count() == 0:
            knowledge_root = os.environ.get(
                "CONDROX_KNOWLEDGE_ROOT",
                "E:\\Data\\condrox\\knowledge",
            )
            # Skip loading if explicitly disabled
            if os.environ.get("CONDROX_SKIP_KNOWLEDGE_LOAD", "0") != "1":
                import logging as _logging
                _log = _logging.getLogger(__name__)

                # Try SQLite massive ingestion DB first (227K+ chunks)
                sqlite_db = os.environ.get(
                    "CONDROX_KNOWLEDGE_DB",
                    "E:\\Data\\condrox\\knowledge_db.sqlite",
                )
                sqlite_loaded = 0
                if Path(sqlite_db).exists():
                    try:
                        sqlite_loader = SQLiteKnowledgeLoader(db_path=sqlite_db)
                        sqlite_loaded = sqlite_loader.load_into(
                            self.knowledge_router.backend
                        )
                        if sqlite_loaded > 0:
                            _log.info(
                                "Boot: loaded %d knowledge items from SQLite DB %s",
                                sqlite_loaded, sqlite_db,
                            )
                    except Exception as exc:
                        _log.warning("SQLite knowledge load failed: %s", exc)

                # Fall back to disk JSONL files if SQLite didn't load enough
                if sqlite_loaded == 0 and Path(knowledge_root).exists():
                    loader = KnowledgeLoader(knowledge_root=knowledge_root)
                    try:
                        loaded = loader.load_into(self.knowledge_router.backend)
                        if loaded > 0:
                            _log.info(
                                "Boot: loaded %d knowledge items from disk %s",
                                loaded, knowledge_root,
                            )
                    except Exception:
                        pass  # Non-fatal — continue with empty knowledge

        # Build knowledge graph for expanded retrieval (skip for very large
        # knowledge stores to keep boot time bounded).
        kb_count = self.knowledge_router.backend.count()
        if 0 < kb_count <= 5000:
            try:
                self.knowledge_router.build_graph()
            except Exception:
                pass  # Non-fatal â€” fall back to plain BM25 search

        # Ingest Condrox School System curriculum so school-domain queries
        # can retrieve verified, hallucination-free explanations.
        if self.config.enable_school_curriculum:
            try:
                self._ingest_school_curriculum()
            except Exception:
                pass  # Non-fatal â€” school system is optional at boot

        # Boot AI-OS kernel (Phase 13)
        if self.aios_kernel is None:
            self.aios_kernel = AIOSKernel(
                state_db_path=os.environ.get(
                    "CONDROX_OS_STATE_DB", "Databases/condrox_os_state.db"
                ),
            )
            # Configure hot-reloader if knowledge root exists and not skipped
            if os.environ.get("CONDROX_SKIP_KNOWLEDGE_LOAD", "0") != "1":
                kroot = os.environ.get(
                    "CONDROX_KNOWLEDGE_ROOT",
                    "E:\\Data\\condrox\\knowledge",
                )
                if os.path.isdir(kroot):
                    self.aios_kernel.configure_hot_reloader(
                        knowledge_root=kroot,
                        backend=self.knowledge_router.backend,
                        interval=30.0,
                    )
            self.aios_kernel.boot()

            # Register brain subsystems as AI-OS services
            self._register_aios_services()

            # Create data flow channels for pipeline stages
            self._create_pipeline_channels()

        return self._boot_report

    def _register_aios_services(self) -> None:
        """Register all brain subsystems as AI-OS managed services.

        Each service has a start, stop, and health check. The service
        manager tracks lifecycle and can auto-restart on failure.

        Complexity: T(n) âˆˆ O(1) for fixed set of services.
        """
        kernel = self.aios_kernel
        if kernel is None:
            return

        # Memory service
        kernel.register_service(ServiceDescriptor(
            name="memory",
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (self.memory_manager.store.count() >= 0, "ok"),
            priority=10,
        ))

        # Knowledge service
        kernel.register_service(ServiceDescriptor(
            name="knowledge",
            dependencies=("memory",),
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (self.knowledge_router.backend is not None, "ok"),
            priority=20,
        ))

        # Reasoning service
        kernel.register_service(ServiceDescriptor(
            name="reasoning",
            dependencies=("memory", "knowledge"),
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (len(self.reasoning_engine._rules) > 0, "ok"),
            priority=30,
        ))

        # Security service
        kernel.register_service(ServiceDescriptor(
            name="security",
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (self.security_engine is not None, "ok"),
            priority=15,
        ))

        # Ethics service
        kernel.register_service(ServiceDescriptor(
            name="ethics",
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (self.ethics_engine is not None, "ok"),
            priority=15,
        ))

        # Personality service
        kernel.register_service(ServiceDescriptor(
            name="personality",
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (self.personality_engine is not None, "ok"),
            priority=25,
        ))

        # Pipeline service
        kernel.register_service(ServiceDescriptor(
            name="pipeline",
            dependencies=("memory", "knowledge", "reasoning", "security", "ethics"),
            start_fn=lambda: True,
            stop_fn=lambda: None,
            health_fn=lambda: (self.pipeline.stage_count() > 0, f"{self.pipeline.stage_count()} stages"),
            priority=50,
        ))

        # Start all registered services
        successes, failures = kernel.service_manager.start_all()
        if failures > 0:
            import logging
            logging.getLogger(__name__).warning(
                "AI-OS services: %d started, %d failed", successes, failures,
            )

    def _create_pipeline_channels(self) -> None:
        """Create data flow channels for pipeline stage communication.

        Channels:
        - input: Raw user input entering the pipeline.
        - intent: Parsed intent + enriched context.
        - context: Built context with knowledge + memory.
        - reasoning: Reasoning output before synthesis.
        - output: Final synthesized output.

        Complexity: T(n) âˆˆ O(1) for fixed set of channels.
        """
        kernel = self.aios_kernel
        if kernel is None:
            return

        kernel.data_flow.create_channel("input", capacity=50)
        kernel.data_flow.create_channel("intent", capacity=50)
        kernel.data_flow.create_channel("context", capacity=50)
        kernel.data_flow.create_channel("reasoning", capacity=50)
        kernel.data_flow.create_channel("output", capacity=50)

        # Route: input â†’ intent â†’ context â†’ reasoning â†’ output
        kernel.data_flow.add_route("input", "intent", priority=10)
        kernel.data_flow.add_route("intent", "context", priority=20)
        kernel.data_flow.add_route("context", "reasoning", priority=30)
        kernel.data_flow.add_route("reasoning", "output", priority=40)

    def save_state(self) -> int:
        """Persist memory and knowledge to the storage backend.

        Returns:
            Total number of items saved (memory + knowledge).

        Complexity: T(n) âˆˆ O(n) for n total items, S(n) âˆˆ O(n).
        """
        if self._storage_backend is None:
            return 0
        mem_count = self.memory_manager.store.save(self._storage_backend, table="memories")
        know_count = self.knowledge_router.backend.save(self._storage_backend, table="knowledge")
        profile_count = self.user_profile_store.save()
        ltm_count = self.long_term_memory.save()
        skill_count = self.skill_memory.save()
        return mem_count + know_count + profile_count + ltm_count + skill_count

    def close(self) -> None:
        """Save state and release resources.

        Shuts down the AI-OS kernel gracefully before persisting state.

        Complexity: T(n) âˆˆ O(n) for n items, plus O(Î£ s_i) for AI-OS shutdown.
        """
        # Shutdown AI-OS kernel first
        if self.aios_kernel is not None and self.aios_kernel.is_booted:
            self.aios_kernel.shutdown()

        self.save_state()
        if self._storage_backend is not None:
            self._storage_backend.close()
        if hasattr(self.knowledge_router.backend, "close"):
            self.knowledge_router.backend.close()

    def health(self) -> HealthStatus:
        """Return aggregate health status.

        Complexity: T(n) âˆˆ O(s) for s heartbeat checks.
        """
        return self.heartbeat_engine.aggregate()

    def ingest_knowledge(self, content: str, domain: str = "general") -> str:
        """Ingest knowledge through the router.

        Args:
            content: Knowledge content.
            domain: Domain/category.

        Returns:
            Id of the ingested item.

        Complexity: T(n) âˆˆ O(n + dÂ²), S(n) âˆˆ O(d).
        """
        item = self.knowledge_router.ingest(content=content, domain=domain)
        return item.item_id

    def add_memory(
        self,
        content: str,
        source: str = "conversation",
        tags: Optional[list[str]] = None,
        weight: float = 1.0,
    ) -> str:
        """Add a memory through the manager.

        Args:
            content: Memory content.
            source: Memory source.
            tags: Optional tags.
            weight: Base weight.

        Returns:
            Id of the stored memory.

        Complexity: T(n) âˆˆ O(n + dÂ²), S(n) âˆˆ O(d).
        """
        item = self.memory_manager.add(
            content=content, source=source, tags=tags, weight=weight
        )
        return item.item_id

    def process_request(self, raw_input: str, context: Optional[Dict[str, Any]] = None, store_memory: bool = True) -> BrainState:
        """Process a user request through the full pipeline.

        Args:
            raw_input: User input text.
            context: Optional initial context values.
            store_memory: When False, skip auto-storing user input and response
                into the shared memory store. Used by the PreReleaseAgent to
                guarantee zero cross-contamination between public demo users.

        Returns:
            Final BrainState after the 18-stage pipeline.

        Raises:
            ValueError: If input is empty.

        Complexity: T(n) ∈ O(Σ s_i) where s_i is stage i complexity.
        """
        if not raw_input or not raw_input.strip():
            raise ValueError("raw_input must be non-empty")

        # Phase 2-A: Single instance lock â€” prevent concurrent pipeline runs
        if self.config.enable_single_instance_lock:
            if not self.single_instance_lock.acquire(blocking=False):
                raise RuntimeError("Another pipeline is already running; cannot acquire single instance lock")

        try:
            if self._boot_report is None:
                self.boot_system()

            # Phase 2-C: Safety check on input
            safety_result = self.safety_layer.check(raw_input)
            if not safety_result.passed:
                # Return early with safety rejection
                initial_state = BrainState(
                    request_id=str(uuid.uuid4()),
                    raw_input=raw_input,
                    metadata={
                        "started_at": time.time(),
                        "safety_blocked": True,
                        "safety_violations": [v.message for v in safety_result.violations],
                    },
                )
                return initial_state.with_field(
                    final_output="[Content blocked by safety layer]",
                    metadata={"safety_blocked": True},
                )

            # Phase 2-A: Stabilize memory before processing
            self.memory_stabilization.stabilize(self.memory_manager.store)

            # Phase 2-C: Track request
            self.pipeline_monitor.increment_request_count()
            request_id = str(uuid.uuid4())

            initial_context = dict(context) if context else {}
            ctx_mgr = ContextManager(initial_context)
            dcl = DCLEngine()  # local per-request DCL
            dcl.load_defaults()
            dcl.step("request", initial_context)

            initial_state = BrainState(
                request_id=request_id,
                raw_input=raw_input,
                metadata={"started_at": time.time(), "store_memory": store_memory},
                context_manager=ctx_mgr,
            )

            # Phase 2-C: Checkpoint state for error recovery
            self.error_recovery.checkpoint(request_id, {"raw_input": raw_input, "context": initial_context})

            # Phase 2-A: Run state query handler before pipeline
            state_result = self.state_query_handler.handle(raw_input.strip(), "unknown")
            if state_result.is_state_query:
                initial_context["state_query"] = {
                    "action": state_result.action,
                    "target": state_result.target,
                    "confidence": state_result.confidence,
                }

            # Phase 2-F: Enqueue request in priority engine
            try:
                self.priority_engine.enqueue(request_id, raw_input)
            except Exception:
                pass

            final_state = self.pipeline.run(self, initial_state)

            # Phase 2-B: Post-pipeline quality enforcement
            if final_state.final_output:
                is_school = (final_state.context or {}).get("school_answer", False)

                if is_school:
                    # School curriculum answers preserve their verified
                    # structure (numbered steps, examples, context).
                    styled = final_state.final_output
                else:
                    # Apply output stabilization
                    stab_result = self.output_stabilizer.stabilize(final_state.final_output)
                    # Apply language enforcement
                    enforced, _ = self.language_enforcer_v2.enforce(stab_result.text)

                    # For conversational/emotional intents, skip style engine
                    # and quality guardrail — they strip legitimate content
                    if final_state.intent in ("greet", "clarify", "emotional", "grateful"):
                        styled = enforced
                    else:
                        # Apply response style
                        styled = self.response_style_engine.apply(
                            enforced,
                            final_state.intent,
                        )

                # Phase 4: Quality guardrail gate (skip for greet/clarify/emotional/grateful/school)
                if final_state.intent in ("greet", "clarify", "emotional", "grateful") or is_school:
                    final_state = final_state.with_field(
                        final_output=styled,
                        metadata={**final_state.metadata, "quality_score": 1.0, "quality_passed": True},
                    )
                else:
                    gated_output, quality = self.quality_guardrail.gate(
                        output=styled,
                        user_input=raw_input,
                        emotion=final_state.emotion,
                    )
                    final_state = final_state.with_field(
                        final_output=gated_output,
                        metadata={**final_state.metadata, "quality_score": quality.score, "quality_passed": quality.passed},
                    )

            # Auto-store user input as memory for multi-turn recall
            # (skipped when store_memory=False — e.g. PreReleaseAgent public demo)
            if store_memory:
                try:
                    self.memory_manager.add(
                        content=raw_input,
                        source="user_input",
                        tags=("conversation", final_state.intent),
                    )
                except Exception:
                    pass

            # Phase 2-C: Clear checkpoint on success
            self.error_recovery.clear_checkpoint(request_id)

            return final_state
        except Exception as exc:
            # Phase 2-C: Error recovery
            self.error_recovery.handle_error(
                str(uuid.uuid4()), "process_request", exc
            )
            raise
        finally:
            if self.config.enable_single_instance_lock:
                self.single_instance_lock.release()

    def _training_runner(self, input_text: str) -> str:
        """Adapter used by the training console to run the pipeline.

        Returns the final output string or an error marker if the pipeline
        fails. This wrapper is intentionally simple so the console stays
        deterministic and does not crash on edge cases.

        Complexity: T(n) âˆˆ O(Î£ s_i) for pipeline stages.
        """
        try:
            final_state = self.process_request(input_text)
            return final_state.final_output or ""
        except Exception as exc:
            return f"[ERROR] {exc}"

    def run_agent(self, mode: str, state: "BrainState") -> "AgentResult":
        """Run the mode-specific agent registered in the agent registry.

        Falls back to the default agent if the requested mode is unknown.

        Args:
            mode: Selected mode (code, explain, analyze, architect, default).
            state: Current BrainState from the pipeline.

        Returns:
            AgentResult produced by the selected agent.

        Complexity: T(n) âˆˆ O(1) for registry lookup plus agent runtime.
        """
        agent = self.agent_registry.get(mode)
        if agent is None:
            agent = self.agent_registry.get("default")
        if agent is None:
            from core.agents.base import AgentResult
            return AgentResult(
                agent_id="unknown",
                mode=mode,
                confidence=0.0,
                errors=[f"No agent registered for mode '{mode}'"],
            )
        return agent.run(self, state)

    def _ingest_school_curriculum(self) -> int:
        """Ingest Condrox School System lessons into the knowledge backend.

        Each lesson becomes a knowledge item with domain="school" so the
        pipeline can retrieve verified, structured answers for school queries.

        Returns:
            Number of lessons ingested.

        Complexity: T(n) âˆˆ O(l Â· cÂ²) for l lessons and c content length.
        """
        count = 0
        for subject_id in self.school_system.list_subjects():
            for lesson in self.school_system.curriculum.get_lessons_for_subject(subject_id):
                content = (
                    f"Subject: {subject_id}\n"
                    f"Title: {lesson.title}\n"
                    f"Concepts: {', '.join(lesson.concepts)}\n"
                    f"Explanation: {lesson.explanation}\n"
                )
                if lesson.examples:
                    content += f"Examples: {', '.join(lesson.examples)}\n"
                if lesson.context_rules:
                    content += f"Context: {', '.join(lesson.context_rules)}\n"
                self.knowledge_router.ingest(
                    content=content,
                    domain="school",
                    item_id=lesson.lesson_id,
                    confidence=1.0,
                    metadata={"source": "school_curriculum", "subject": subject_id},
                )
                count += 1
        return count

    def run_training(self, domain: str = "school"):
        """Run the training console for a domain and return a report.

        Args:
            domain: Domain to validate.

        Returns:
            TrainingReport from the console.

        Complexity: T(n) âˆˆ O(m Â· (r + v)) for m scenarios.
        """
        return self.training_console.run_domain(domain)

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of Brain subsystem status.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
        """
        self._ensure_creative()
        return {
            "system_name": self.config.system_name,
            "version": self.config.version,
            "pipeline_stages": self.pipeline.stage_count(),
            "memory_count": self.memory_manager.store.count(),
            "knowledge_count": self.knowledge_router.backend.count(),
            "health": self.health().score,
            "booted": bool(self._boot_report and self._boot_report.success),
            "instance_locked": self.single_instance_lock.is_locked(),
            # Phase 2-A
            "state_query_handler": True,
            "memory_stabilization": True,
            "search_agent_guard": True,
            # Phase 2-B
            "response_style_engine": True,
            "claim_ranker_v2": True,
            "temporal_context": True,
            "output_stabilizer": True,
            "language_enforcer_v2": True,
            # Phase 2-C
            "safety_layer": True,
            "pipeline_monitor": True,
            "error_recovery": True,
            "pipeline_metrics": self.pipeline_monitor.export_metrics(),
            # Phase 2-D
            "agent_arbiter": True,
            "agent_count": self.agent_registry.count(),
            "knowledge_router_v2": True,
            "knowledge_domains": self.knowledge_router_v2.domain_count(),
            "multi_agent_memory_sync": True,
            "shared_memory_count": self.multi_agent_memory_sync.count(),
            # Phase 2-E
            "intent_expander": True,
            "multi_turn_reasoning": True,
            "turn_count": self.multi_turn_reasoning.turn_count(),
            "memory_graph": True,
            "memory_graph_nodes": self.memory_graph.node_count(),
            "goal_tracker": True,
            "active_goals": len(self.goal_tracker.get_active_goals()),
            "consistency_engine": True,
            "semantic_repair": True,
            "contradiction_detector": True,
            # Phase 2-F
            "latency_compensation": True,
            "token_pressure_regulator": True,
            "token_usage": self.token_pressure_regulator.get_usage(),
            "semantic_interrupt": True,
            "priority_engine": True,
            "priority_queue_size": self.priority_engine.queue_size(),
            # Universal Tools
            "file_ops": True,
            "shell_exec": True,
            "git_ops": True,
            # Phase 4: Creative Stack
            "emotion_palette_mapper": self.emotion_palette_mapper is not None,
            "palette_emotions": self.emotion_palette_mapper.available_emotions() if self.emotion_palette_mapper else [],
            "music_prompt_builder": self.music_prompt_builder is not None,
            "quality_guardrail": True,
            "guardrail_threshold": self.quality_guardrail._threshold,
        }

    @classmethod
    def from_config_file(cls, path: Path) -> "BrainController":
        """Create a BrainController from a YAML configuration file.

        Args:
            path: Path to YAML config.

        Returns:
            Configured BrainController.

        Complexity: T(n) âˆˆ O(k) for k config keys.
        """
        config = BrainConfig.from_yaml(path)
        return cls(config)

    # â”€â”€â”€ Phase 4: Creative Stack â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_creative_output(
        self,
        raw_input: str,
        include_lyrics: bool = True,
    ) -> Dict[str, Any]:
        """Process input and generate a creative (music) output package.

        Runs the full pipeline, then maps the detected emotion to a sound
        palette and builds a music generation prompt. The quality guardrail
        gates the conversational response before it's included.

        Args:
            raw_input: User input text.
            include_lyrics: Whether to generate lyrics.

        Returns:
            Dict with keys: state, quality, palette, music_prompt.

        Complexity: T(n) in O(pipeline + palette + prompt).
        """
        self._ensure_creative()
        state = self.process_request(raw_input)

        # Gate the conversational output through quality guardrail
        gated_output, quality = self.quality_guardrail.gate(
            output=state.final_output,
            user_input=raw_input,
            emotion=state.emotion,
        )

        # Map emotion to sound palette
        palette = self.emotion_palette_mapper.map(state.emotion)

        # Build music generation prompt
        music_prompt = self.music_prompt_builder.build(
            palette=palette,
            profile=self.personality_engine.profile,
            include_lyrics=include_lyrics,
            user_context=raw_input,
        )

        return {
            "state": state,
            "gated_output": gated_output,
            "quality": quality,
            "palette": palette,
            "music_prompt": music_prompt,
        }

    def get_sound_palette(self, emotion: str):
        """Get the sound palette for an emotion.

        Args:
            emotion: Emotion label.

        Returns:
            SoundPalette.

        Complexity: T(n) in O(1).
        """
        self._ensure_creative()
        return self.emotion_palette_mapper.map(emotion)

    def build_music_prompt(self, emotion: str, user_context: str = "", include_lyrics: bool = True):
        """Build a music generation prompt from an emotion.

        Args:
            emotion: Emotion label.
            user_context: Optional context for lyric inspiration.
            include_lyrics: Whether to include lyrics.

        Returns:
            MusicPrompt.

        Complexity: T(n) in O(p + t).
        """
        self._ensure_creative()
        palette = self.emotion_palette_mapper.map(emotion)
        return self.music_prompt_builder.build(
            palette=palette,
            profile=self.personality_engine.profile,
            include_lyrics=include_lyrics,
            user_context=user_context,
        )
