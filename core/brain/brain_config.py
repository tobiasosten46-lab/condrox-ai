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
Brain configuration management.

Provides immutable, validated configuration objects for the BrainController
and its subsystems.

Mathematical Foundation:
- Validation complexity: T(n) âˆˆ O(k) where k is the number of config keys.
- Memory usage: S(n) âˆˆ O(1) for a fixed-size configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from core import __version__ as _CORE_VERSION
except ImportError:
    _CORE_VERSION = "3.4.0"


@dataclass(frozen=True, slots=True)
class BrainConfig:
    """
    Immutable configuration for the unified Brain system.

    Attributes:
        system_name: Human-readable system identifier.
        version: Semantic version string.
        environment: Runtime environment (development, staging, production).
        log_level: Logging verbosity.
        max_input_length: Maximum allowed raw input length in characters.
        default_profile: Active personality profile.
        enable_learning: Whether meta-learning updates are enabled.
        memory_decay_halflife_seconds: Half-life for memory relevance decay.
        blocked_patterns: Compiled security patterns (stored as strings).
        blocked_topics: Ethics blocked topics.
        metadata: Additional free-form configuration.

    Complexity: T(load) âˆˆ O(k), S(load) âˆˆ O(1) where k is config key count.
    """

    system_name: str = "Condrox AI"
    version: str = _CORE_VERSION
    environment: str = "development"
    log_level: str = "INFO"
    max_input_length: int = 4096
    default_profile: str = "condrox"
    enable_learning: bool = True
    enable_single_instance_lock: bool = True
    enable_school_curriculum: bool = True
    persistence_path: Optional[str] = None
    use_disk_knowledge: bool = False
    memory_decay_halflife_seconds: float = 86400.0
    blocked_patterns: tuple[str, ...] = field(default_factory=tuple)
    blocked_topics: tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)

    @staticmethod
    def _validate_positive(name: str, value: int) -> int:
        """Validate that an integer value is strictly positive.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def __post_init__(self) -> None:
        """Run post-construction validations.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
        """
        object.__setattr__(
            self, "max_input_length", self._validate_positive("max_input_length", self.max_input_length)
        )
        object.__setattr__(
            self,
            "memory_decay_halflife_seconds",
            self._validate_positive(
                "memory_decay_halflife_seconds", int(self.memory_decay_halflife_seconds)
            ),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "BrainConfig":
        """Load configuration from a YAML file.

        Args:
            path: Filesystem path to the YAML config file.

        Returns:
            A validated BrainConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is malformed or missing required keys.

        Complexity: T(n) âˆˆ O(k) where k is the number of keys, S(n) âˆˆ O(1).
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = yaml.safe_load(handle) or {}

        system = data.get("system", {})
        brain = data.get("brain", {})
        security = data.get("security", {})
        ethics = data.get("ethics", {})

        return cls(
            system_name=system.get("name", cls.system_name),
            version=system.get("version", cls.version),
            environment=system.get("environment", cls.environment),
            log_level=system.get("log_level", cls.log_level),
            max_input_length=brain.get("max_input_length", cls.max_input_length),
            default_profile=brain.get("default_profile", cls.default_profile),
            enable_learning=brain.get("enable_learning", cls.enable_learning),
            enable_single_instance_lock=brain.get("enable_single_instance_lock", cls.enable_single_instance_lock),
            enable_school_curriculum=brain.get("enable_school_curriculum", cls.enable_school_curriculum),
            persistence_path=brain.get("persistence_path", cls.persistence_path),
            use_disk_knowledge=brain.get("use_disk_knowledge", cls.use_disk_knowledge),
            memory_decay_halflife_seconds=brain.get(
                "memory_decay_halflife_seconds", cls.memory_decay_halflife_seconds
            ),
            blocked_patterns=tuple(security.get("blocked_patterns", [])),
            blocked_topics=tuple(ethics.get("blocked_topics", [])),
            metadata=data,
        )

    @classmethod
    def default(cls) -> "BrainConfig":
        """Return the default production-ready configuration.

        Complexity: T(n) âˆˆ O(1), S(n) âˆˆ O(1).
        """
        return cls()
