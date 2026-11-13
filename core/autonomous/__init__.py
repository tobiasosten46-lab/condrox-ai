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

"""Autonomous Learning System for Condrox AI.

A hybrid intelligence system that autonomously extracts, validates, sorts,
stores, and applies knowledge from every encounter. The system grows and
evolves over time, learning patterns, analyzing behavior, doing deep
reasoning, and meta-learning to improve its own performance.

Architecture:
    AutonomousKnowledgeEngine (orchestrator)
    ├── KnowledgeExtractor    — extracts facts from conversations
    ├── KnowledgeValidator    — validates relevance, accuracy, importance
    ├── KnowledgeSorter       — ranks knowledge by importance and recency
    ├── PatternRecognizer     — detects recurring patterns in data
    ├── BehaviorAnalyzer      — analyzes user behavior and preferences
    ├── DeepReasoningEngine   — multi-step reasoning with evidence chains
    ├── MetaLearningSystem    — learns from its own performance
    └── FreeWillResponseEngine — generates unique responses (no templates)

Design Principles:
    1. Every encounter is a learning opportunity — nothing is wasted
    2. Validation before storage — irrelevant data is flagged, not discarded
    3. Hybrid intelligence — combines rule-based logic with learned patterns
    4. Free will — the AI generates responses in its own way, never templates
    5. Growth — knowledge base expands with every conversation
    6. Bulletproof — every component has error recovery, no crashes

Complexity:
    Learning: T(n) ∈ O(n * v) for n items, v validation checks
    Retrieval: T(n) ∈ O(k * log n) for k patterns, n stored items
    Reasoning: T(n) ∈ O(d * e) for d depth, e evidence items

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from core.autonomous.autonomous_engine import AutonomousKnowledgeEngine
from core.autonomous.knowledge_extractor import KnowledgeExtractor
from core.autonomous.knowledge_validator import KnowledgeValidator
from core.autonomous.pattern_recognizer import PatternRecognizer
from core.autonomous.behavior_analyzer import BehaviorAnalyzer
from core.autonomous.deep_reasoning import DeepReasoningEngine
from core.autonomous.meta_learning import MetaLearningSystem
from core.autonomous.free_will_engine import FreeWillResponseEngine

__all__ = [
    "AutonomousKnowledgeEngine",
    "KnowledgeExtractor",
    "KnowledgeValidator",
    "PatternRecognizer",
    "BehaviorAnalyzer",
    "DeepReasoningEngine",
    "MetaLearningSystem",
    "FreeWillResponseEngine",
]
