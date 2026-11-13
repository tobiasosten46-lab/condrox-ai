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

"""Cognitive Brain — Synthetic Cognitive Architecture v1.0.

Dette er den syntetiske hjernen. Ikke moduler. Ikke simulering.
Én sammenhengende kognitiv strøm der alt henger sammen:

    reward → bygger erfaring
    erfaring → bygger mening
    mening → bygger prioritet
    prioritet → bygger mål
    mål → bygger visjon
    visjon → styrer pipeline
    pipeline → styrer hele hjernen

Komponenter:
    CognitiveBrainCore — Den unified orchestrator (én hjerne)
    RewardSystem — Kognitiv verdi (stability, insight, clarity, strategy, breakthrough)
    GoalSystem — Kognitiv retning (understand, improve, clarify, connect, master)
    VisionSystem — Syntetisk identitet (systemets "hvorfor")
    OpinionSystem — Erfaringens vurdering (important, unclear, working, solid)
    CognitivePhases — Syntetiske hjernefaser (reasoning, pattern, analysis, reverse, structure)
    ExperienceMemory — Syntetisk erfaring (kognitiv struktur, ikke bare data)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from core.cognitive_brain.brain_core import (
    CognitiveBrainCore,
    BrainState,
    ThoughtResult,
)
from core.cognitive_brain.unified_pipeline import (
    UnifiedCognitivePipeline,
    PipelineContext,
)
from core.cognitive_brain.reward_system import (
    RewardSystem,
    Reward,
    FactValidationRecord,
)
from core.cognitive_brain.goal_system import (
    GoalSystem,
    Goal,
)
from core.cognitive_brain.vision_system import (
    VisionSystem,
    VisionDirective,
    AlignmentResult,
)
from core.cognitive_brain.opinion_system import (
    OpinionSystem,
    Opinion,
)
from core.cognitive_brain.cognitive_phases import (
    CognitivePhases,
    ReasoningPhase,
    PatternPhase,
    AnalysisPhase,
    ReversePhase,
    StructurePhase,
    PhaseResult,
    CognitiveFlowResult,
)
from core.cognitive_brain.experience_memory import (
    ExperienceMemory,
    Experience,
)

__all__ = [
    # Core
    "CognitiveBrainCore",
    "BrainState",
    "ThoughtResult",
    # Unified Pipeline
    "UnifiedCognitivePipeline",
    "PipelineContext",
    # Reward
    "RewardSystem",
    "Reward",
    "FactValidationRecord",
    # Goals
    "GoalSystem",
    "Goal",
    # Vision
    "VisionSystem",
    "VisionDirective",
    "AlignmentResult",
    # Opinions
    "OpinionSystem",
    "Opinion",
    # Cognitive Phases
    "CognitivePhases",
    "ReasoningPhase",
    "PatternPhase",
    "AnalysisPhase",
    "ReversePhase",
    "StructurePhase",
    "PhaseResult",
    "CognitiveFlowResult",
    # Experience Memory
    "ExperienceMemory",
    "Experience",
]
