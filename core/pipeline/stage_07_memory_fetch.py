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
Stage 07: Memory Fetch.

Retrieves semantically relevant memories using the MemoryManager's vector
search. Access counts are incremented for retrieved items.

Mathematical Foundation:
- Top-k memory search is O(n log k) for n stored memories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage07MemoryFetch:
    """Real memory retrieval stage."""

    __slots__ = ("_top_k", "_query_expansions")

    def __init__(self, top_k: int = 5) -> None:
        """Initialize with a default top-k.

        Args:
            top_k: Maximum memory items to retrieve.
        """
        self._top_k = max(1, top_k)
        self._query_expansions = {
            "what's my name": ["name", "user's name", "called"],
            "what is my name": ["name", "user's name", "called"],
            "what do i do": ["profession", "work", "job", "user's profession"],
            "what is my job": ["profession", "work", "job", "user's profession"],
            "what is my profession": ["profession", "work", "job", "user's profession"],
        }

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run memory retrieval with graph and sync integration.

        Complexity: T(n) ∈ O(n log k + g + s), S(n) ∈ O(k).
        """
        # Check for entity-specific queries and do direct tag lookup
        entity_type = self._detect_entity_query(state.processed_input)
        
        if entity_type:
            # Try tag-based retrieval first
            try:
                query_embedding = controller.memory_manager.embed(state.processed_input)
                items = controller.memory_manager.store.search(
                    query_embedding=query_embedding,
                    top_k=self._top_k,
                    tag_filter=entity_type,
                )
                print(f"[Stage07MemoryFetch] Tag-based retrieval for '{entity_type}': {len(items)} items")
            except Exception as e:
                print(f"[Stage07MemoryFetch] Tag-based retrieval failed: {e}")
                items = controller.memory_manager.search(
                    query=state.processed_input,
                    top_k=self._top_k,
                )
        else:
            # Expand query for entity questions
            query = self._expand_query(state.processed_input)
            items = controller.memory_manager.search(
                query=query,
                top_k=self._top_k,
            )

        # Increase access counts for retrieved items
        for item in items:
            controller.memory_manager.store.add(item.with_access())

        ids: List[str] = [item.item_id for item in items]
        memory_texts = [item.content for item in items]

        # Also retrieve user pattern memories for personalization
        # These store learned behavioral/emotional patterns about the user
        try:
            pattern_items = controller.memory_manager.store.search(
                query_embedding=controller.memory_manager.embed(state.processed_input),
                top_k=3,
                tag_filter="user_pattern",
            )
            for p_item in pattern_items:
                if p_item.item_id not in ids and p_item.content not in memory_texts:
                    memory_texts.append(p_item.content)
                    ids.append(p_item.item_id)
        except Exception:
            pass

        # Search memory graph for related nodes
        try:
            graph_results = controller.memory_graph.search(
                state.processed_input, top_k=self._top_k
            )
            if graph_results:
                # graph_results is List[Tuple[MemoryNode, float]]
                for node, score in graph_results:
                    if hasattr(node, "content") and node.content and node.content not in memory_texts:
                        memory_texts.append(node.content)
                        ids.append(node.node_id)
        except Exception:
            pass

        # Sync memory with multi-agent shared buffer
        try:
            controller.multi_agent_memory_sync.write(
                key=state.request_id,
                value=memory_texts,
                agent_id="pipeline",
            )
        except Exception:
            pass

        cm = state.context_manager if state.context_manager is not None else controller.context_manager
        cm.set("has_memory", bool(ids))
        cm.set("memory_count", len(ids))
        cm.set("memory_texts", memory_texts)

        return state.with_field(
            retrieved_memories=ids,
            context=cm.current(),
        )

    def _expand_query(self, query: str) -> str:
        """Expand query with entity-specific keywords for better memory retrieval.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        query_lower = query.lower().strip()
        
        for pattern, expansions in self._query_expansions.items():
            if pattern in query_lower:
                # Add expansions to query
                expanded = f"{query} {' '.join(expansions)}"
                return expanded
        
        return query

    def _detect_entity_query(self, query: str) -> Optional[str]:
        """Detect if query is asking about a specific entity type.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        query_lower = query.lower().strip()
        
        entity_keywords = {
            "name": ["name", "called", "who am i"],
            "profession": ["do i do", "job", "work", "profession"],
        }
        
        for entity_type, keywords in entity_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return entity_type
        
        return None
