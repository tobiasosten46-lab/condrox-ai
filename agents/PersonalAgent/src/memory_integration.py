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
Memory Integration Layer — Integrates agent functions with Main AI memory.

Functions:
- TaskMemory (store tasks in Main AI memory)
- ReminderMemory (store reminders)
- NoteMemory (store notes)
- ContextLinking (link PA data to user's identity)
- Multi-turn retention (PA remembers tasks over time)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

from agents.PersonalAgent.src.capability_layer import Task, Reminder, Note


class MemoryIntegration:
    """
    Memory integration for Personal Assistant.

    Integrates PA data (tasks, reminders, notes) with Main AI memory.
    Ensures PA data is linked to user's identity and persists across conversations.

    Complexity:
        store_task: T(n) ∈ O(m) for m memory operations
        retrieve_tasks: T(n) ∈ O(k) for k memory items
    """

    __slots__ = ("_brain_controller", "_memory_prefix")

    def __init__(self, brain_controller: "BrainController") -> None:
        """Initialize MemoryIntegration.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._brain_controller = brain_controller
        self._memory_prefix = "pa_"  # Prefix for PA-specific memory

    def store_task(self, task: Task) -> bool:
        """Store task in Main AI memory.

        Complexity: T(n) ∈ O(m) for m memory operations.
        """
        try:
            # Create memory key
            memory_key = f"{self._memory_prefix}task_{task.id}"
            
            # Serialize task data
            task_data = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "priority": task.priority.value,
                "completed": task.completed,
                "created_at": task.created_at,
                "due_at": task.due_at,
            }
            
            # Store in Main AI memory
            if hasattr(self._brain_controller, 'memory_manager'):
                self._brain_controller.memory_manager.store(
                    key=memory_key,
                    value=json.dumps(task_data),
                    tags=["task", "pa", task.priority.value],
                )
                return True
        except Exception:
            pass
        return False

    def retrieve_tasks(self, completed: Optional[bool] = None) -> List[Task]:
        """Retrieve tasks from Main AI memory.

        Complexity: T(n) ∈ O(k) for k memory items.
        """
        tasks = []
        try:
            if hasattr(self._brain_controller, 'memory_manager'):
                # Retrieve all PA task memories
                memories = self._brain_controller.memory_manager.retrieve(
                    query="pa_task",
                    top_k=100,
                )
                
                for memory in memories:
                    try:
                        task_data = json.loads(memory.value)
                        # Filter by completion status if specified
                        if completed is None or task_data.get("completed") == completed:
                            from agents.PersonalAgent.src.capability_layer import TaskPriority
                            task = Task(
                                id=task_data["id"],
                                title=task_data["title"],
                                description=task_data.get("description", ""),
                                priority=TaskPriority(task_data.get("priority", "medium")),
                                completed=task_data.get("completed", False),
                                created_at=task_data.get("created_at", ""),
                                due_at=task_data.get("due_at"),
                            )
                            tasks.append(task)
                    except Exception:
                        continue
        except Exception:
            pass
        return tasks

    def store_reminder(self, reminder: Reminder) -> bool:
        """Store reminder in Main AI memory.

        Complexity: T(n) ∈ O(m) for m memory operations.
        """
        try:
            memory_key = f"{self._memory_prefix}reminder_{reminder.id}"
            
            reminder_data = {
                "id": reminder.id,
                "message": reminder.message,
                "remind_at": reminder.remind_at,
                "created_at": reminder.created_at,
            }
            
            if hasattr(self._brain_controller, 'memory_manager'):
                self._brain_controller.memory_manager.store(
                    key=memory_key,
                    value=json.dumps(reminder_data),
                    tags=["reminder", "pa"],
                )
                return True
        except Exception:
            pass
        return False

    def retrieve_reminders(self) -> List[Reminder]:
        """Retrieve reminders from Main AI memory.

        Complexity: T(n) ∈ O(k) for k memory items.
        """
        reminders = []
        try:
            if hasattr(self._brain_controller, 'memory_manager'):
                memories = self._brain_controller.memory_manager.retrieve(
                    query="pa_reminder",
                    top_k=100,
                )
                
                for memory in memories:
                    try:
                        reminder_data = json.loads(memory.value)
                        reminder = Reminder(
                            id=reminder_data["id"],
                            message=reminder_data["message"],
                            remind_at=reminder_data["remind_at"],
                            created_at=reminder_data.get("created_at", ""),
                        )
                        reminders.append(reminder)
                    except Exception:
                        continue
        except Exception:
            pass
        return reminders

    def store_note(self, note: Note) -> bool:
        """Store note in Main AI memory.

        Complexity: T(n) ∈ O(m) for m memory operations.
        """
        try:
            memory_key = f"{self._memory_prefix}note_{note.id}"
            
            note_data = {
                "id": note.id,
                "content": note.content,
                "created_at": note.created_at,
            }
            
            if hasattr(self._brain_controller, 'memory_manager'):
                self._brain_controller.memory_manager.store(
                    key=memory_key,
                    value=json.dumps(note_data),
                    tags=["note", "pa"],
                )
                return True
        except Exception:
            pass
        return False

    def retrieve_notes(self) -> List[Note]:
        """Retrieve notes from Main AI memory.

        Complexity: T(n) ∈ O(k) for k memory items.
        """
        notes = []
        try:
            if hasattr(self._brain_controller, 'memory_manager'):
                memories = self._brain_controller.memory_manager.retrieve(
                    query="pa_note",
                    top_k=100,
                )
                
                for memory in memories:
                    try:
                        note_data = json.loads(memory.value)
                        note = Note(
                            id=note_data["id"],
                            content=note_data["content"],
                            created_at=note_data.get("created_at", ""),
                        )
                        notes.append(note)
                    except Exception:
                        continue
        except Exception:
            pass
        return notes

    def link_to_user_identity(self, user_id: str) -> bool:
        """Link PA data to user's identity in Main AI memory.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        try:
            memory_key = f"{self._memory_prefix}user_id"
            
            if hasattr(self._brain_controller, 'memory_manager'):
                self._brain_controller.memory_manager.store(
                    key=memory_key,
                    value=user_id,
                    tags=["user", "identity", "pa"],
                )
                return True
        except Exception:
            pass
        return False

    def get_user_identity(self) -> Optional[str]:
        """Get user's identity from Main AI memory.

        Complexity: T(n) ∈ O(k) for k memory items.
        """
        try:
            if hasattr(self._brain_controller, 'memory_manager'):
                memories = self._brain_controller.memory_manager.retrieve(
                    query="pa_user_id",
                    top_k=1,
                )
                
                if memories:
                    return memories[0].value
        except Exception:
            pass
        return None

    def clear_pa_memory(self) -> bool:
        """Clear all PA-specific memory from Main AI.

        Complexity: T(n) ∈ O(k) for k memory items.
        """
        try:
            if hasattr(self._brain_controller, 'memory_manager'):
                # Retrieve all PA memories
                memories = self._brain_controller.memory_manager.retrieve(
                    query="pa_",
                    top_k=1000,
                )
                
                # Delete each PA memory
                for memory in memories:
                    try:
                        self._brain_controller.memory_manager.delete(memory.key)
                    except Exception:
                        continue
                return True
        except Exception:
            pass
        return False
