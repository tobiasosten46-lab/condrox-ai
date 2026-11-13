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

"""Conversational intelligence components for the unified Brain system.

Phase 2 components that extend the 14-stage pipeline with stabilization,
coordination, and quality layers.
"""

# Phase 2-A: Critical Stabilization
from core.conversational.state_query_handler import StateQueryHandler, StateQueryResult
from core.conversational.single_instance_lock import SingleInstanceLock
from core.conversational.memory_stabilization import (
    MemoryStabilizationLayer,
    MemoryPrioritizer,
    MemoryFlushPolicy,
    MemoryCompressor,
    MemoryFallbackPolicy,
)
from core.conversational.search_agent_guard import SearchAgentGuard

# Phase 2-B: Conversational Quality
from core.conversational.response_style_engine import (
    ResponseStyleEngine,
    StyleProfile,
)
from core.conversational.claim_ranker_v2 import (
    ClaimRankerV2,
    Claim,
    RankedClaim,
)
from core.conversational.temporal_context import (
    TemporalContextLayer,
    TemporalEntry,
)
from core.conversational.output_stabilizer import (
    OutputStabilizer,
    StabilizationResult,
)
from core.conversational.language_enforcer_v2 import (
    LanguageEnforcerV2,
    PersonaRules,
)

# Phase 2-C: Observability & Safety
from core.conversational.safety_layer import (
    SafetyLayer,
    SafetyLevel,
    SafetyResult,
    SafetyRule,
    SafetyViolation,
)
from core.conversational.pipeline_monitor import (
    PipelineMonitor,
    StageSpan,
    StageMetrics,
)
from core.conversational.error_recovery import (
    ErrorRecoveryLayer,
    ErrorRecord,
    RecoveryResult,
    RecoveryType,
)

# Phase 2-D: Agent Coordination
from core.conversational.agent_arbiter import (
    AgentArbiter,
    AgentCapability,
    AgentDescriptor,
    ArbitrationResult,
)
from core.conversational.knowledge_router_v2 import (
    KnowledgeRouterV2,
    DomainConfig,
    RoutingDecision,
)
from core.conversational.multi_agent_memory_sync import (
    MultiAgentMemorySync,
    SyncEntry,
    SyncResult,
)

# Phase 2-E: Advanced Conversational Intelligence
from core.conversational.intent_expander import (
    SemanticIntentExpander,
    ExpandedIntent,
)
from core.conversational.multi_turn_reasoning import (
    MultiTurnReasoningEngine,
    TurnRecord,
    ReasoningContext,
)
from core.conversational.memory_graph import (
    ConversationalMemoryGraph,
    MemoryNode,
    GraphEdge,
)
from core.conversational.goal_tracker import (
    DialogueGoalTracker,
    Goal,
    GoalStatus,
)
from core.conversational.consistency_engine import (
    ConsistencyEngine,
    ConsistencyReport,
)
from core.conversational.semantic_repair import (
    SemanticRepair,
    RepairResult,
)
from core.conversational.contradiction_detector import (
    ContradictionDetector,
    ContradictionFinding,
    ContradictionReport,
)

# Phase 2-F: System-Level Properties
from core.conversational.latency_compensation import (
    LatencyCompensationLayer,
    LatencyMeasurement,
    LatencyPrediction,
)
from core.conversational.token_pressure_regulator import (
    TokenPressureRegulator,
    TokenBudget,
    PressureReport,
)
from core.conversational.semantic_interrupt import (
    SemanticInterruptHandler,
    InterruptType,
    InterruptResult,
)
from core.conversational.priority_engine import (
    ConversationalPriorityEngine,
    PriorityTier,
    PriorityScore,
    PrioritizedItem,
)

# Conversational Engine (follow-up questions, curiosity, topic tracking)
from core.conversational.conversational_engine import (
    ConversationalEngine,
    TopicTracker,
    FollowUpGenerator,
    CuriosityEngine,
    ConversationProfile,
    TrackedTopic,
)

__all__ = [
    # Phase 2-A
    "StateQueryHandler",
    "StateQueryResult",
    "SingleInstanceLock",
    "MemoryStabilizationLayer",
    "MemoryPrioritizer",
    "MemoryFlushPolicy",
    "MemoryCompressor",
    "MemoryFallbackPolicy",
    "SearchAgentGuard",
    # Phase 2-B
    "ResponseStyleEngine",
    "StyleProfile",
    "ClaimRankerV2",
    "Claim",
    "RankedClaim",
    "TemporalContextLayer",
    "TemporalEntry",
    "OutputStabilizer",
    "StabilizationResult",
    "LanguageEnforcerV2",
    "PersonaRules",
    # Phase 2-C
    "SafetyLayer",
    "SafetyLevel",
    "SafetyResult",
    "SafetyRule",
    "SafetyViolation",
    "PipelineMonitor",
    "StageSpan",
    "StageMetrics",
    "ErrorRecoveryLayer",
    "ErrorRecord",
    "RecoveryResult",
    "RecoveryType",
    # Phase 2-D
    "AgentArbiter",
    "AgentCapability",
    "AgentDescriptor",
    "ArbitrationResult",
    "KnowledgeRouterV2",
    "DomainConfig",
    "RoutingDecision",
    "MultiAgentMemorySync",
    "SyncEntry",
    "SyncResult",
    # Phase 2-E
    "SemanticIntentExpander",
    "ExpandedIntent",
    "MultiTurnReasoningEngine",
    "TurnRecord",
    "ReasoningContext",
    "ConversationalMemoryGraph",
    "MemoryNode",
    "GraphEdge",
    "DialogueGoalTracker",
    "Goal",
    "GoalStatus",
    "ConsistencyEngine",
    "ConsistencyReport",
    "SemanticRepair",
    "RepairResult",
    "ContradictionDetector",
    "ContradictionFinding",
    "ContradictionReport",
    # Phase 2-F
    "LatencyCompensationLayer",
    "LatencyMeasurement",
    "LatencyPrediction",
    "TokenPressureRegulator",
    "TokenBudget",
    "PressureReport",
    "SemanticInterruptHandler",
    "InterruptType",
    "InterruptResult",
    "ConversationalPriorityEngine",
    "PriorityTier",
    "PriorityScore",
    "PrioritizedItem",
    # Conversational Engine
    "ConversationalEngine",
    "TopicTracker",
    "FollowUpGenerator",
    "CuriosityEngine",
    "ConversationProfile",
    "TrackedTopic",
]
