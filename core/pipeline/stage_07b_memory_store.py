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
Stage 07b: Memory Store.

Stores important conversation context to memory for multi-turn retention.
Extracts entities like name, profession, and preferences from user input.

Complexity: T(n) in O(n) where n = input length, S(n) in O(1).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage07bMemoryStore:
    """Store important conversation context to memory."""

    __slots__ = ("_entity_patterns",)

    def __init__(self) -> None:
        """Initialize with entity extraction patterns."""
        self._entity_patterns = {
            "name": [
                r"my name is (\w+)",
                r"call me (\w+)",
            ],
            "profession": [
                r"i work as a? (\w+(?: \w+)*)",
                r"i am a? (\w+(?: \w+)*)",
            ],
        }

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Extract and store important entities to memory.

        Complexity: T(n) in O(n) where n = input length, S(n) in O(1).
        """
        user_input_lower = state.processed_input.lower()
        
        # Extract entities
        entities = self._extract_entities(user_input_lower)
        
        # Store each entity as a memory with multiple variations for better retrieval
        for entity_type, entity_value in entities:
            # Store with multiple phrasings for better retrieval
            variations = [
                f"{entity_type}: {entity_value}",
                f"User's {entity_type} is {entity_value}",
                f"The user is a {entity_value}" if entity_type == "profession" else None,
                f"The user's name is {entity_value}" if entity_type == "name" else None,
            ]
            
            for memory_content in variations:
                if memory_content is None:
                    continue
                    
                tags = [entity_type, "conversation", "entity"]
                
                try:
                    item_id = controller.add_memory(
                        content=memory_content,
                        source="conversation",
                        tags=tags,
                        weight=1.0,
                    )
                    print(f"[Stage07bMemoryStore] Stored memory: {memory_content} (id: {item_id})")
                except Exception as e:
                    print(f"[Stage07bMemoryStore] Failed to store memory: {e}")
        
        return state

    def _extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract entities from text using regex patterns.

        Complexity: T(n) in O(n * p) where p = patterns, S(n) in O(1).
        """
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entity_value = match.group(1).strip()
                    if entity_value:
                        entities.append((entity_type, entity_value))
                        break  # Use first match for each entity type
        
        return entities
