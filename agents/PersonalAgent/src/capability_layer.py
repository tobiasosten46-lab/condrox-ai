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
Capability Layer — Core Tools for Personal Assistant Agent.

Provides the tools that the PA role requires:
- TaskManager (store, retrieve, list tasks)
- ReminderEngine (timed reminders)
- InfoLookup (quick answers to facts, definitions, simple questions)
- NoteStore (store short notes)
- QuickPlan (create simple plans, steps, overviews)
- ToneController (PA tone, polite, structured)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class TaskPriority(Enum):
    """Task priority levels.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class Task:
    """
    Task data structure.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    id: str
    title: str
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    due_at: Optional[str] = None


class TaskManager:
    """
    Task management tool for Personal Assistant.

    Complexity:
        add: T(n) ∈ O(1), S(n) ∈ O(1)
        get: T(n) ∈ O(1), S(n) ∈ O(1)
        list: T(n) ∈ O(n), S(n) ∈ O(n) for n tasks
    """

    __slots__ = ("_tasks",)

    def __init__(self) -> None:
        """Initialize TaskManager.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._tasks: Dict[str, Task] = {}

    def add(self, title: str, description: str = "", priority: TaskPriority = TaskPriority.MEDIUM) -> Task:
        """Add a new task.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        task_id = uuid.uuid4().hex[:16]
        task = Task(id=task_id, title=title, description=description, priority=priority)
        object.__setattr__(self, "_tasks", {**self._tasks, task_id: task})
        return task

    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._tasks.get(task_id)

    def list(self, completed: Optional[bool] = None) -> List[Task]:
        """List tasks, optionally filtered by completion status.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n tasks.
        """
        tasks = list(self._tasks.values())
        if completed is not None:
            tasks = [t for t in tasks if t.completed == completed]
        return tasks

    def complete(self, task_id: str) -> bool:
        """Mark a task as completed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        # Create new completed task
        completed_task = Task(
            id=task.id,
            title=task.title,
            description=task.description,
            priority=task.priority,
            completed=True,
            created_at=task.created_at,
            due_at=task.due_at,
        )
        object.__setattr__(self, "_tasks", {**self._tasks, task_id: completed_task})
        return True

    def delete(self, task_id: str) -> bool:
        """Delete a task.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if task_id in self._tasks:
            object.__setattr__(self, "_tasks", {k: v for k, v in self._tasks.items() if k != task_id})
            return True
        return False


@dataclass(frozen=True, slots=True)
class Reminder:
    """
    Reminder data structure.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    id: str
    message: str
    remind_at: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ReminderEngine:
    """
    Reminder engine for Personal Assistant.

    Complexity:
        add: T(n) ∈ O(1), S(n) ∈ O(1)
        list: T(n) ∈ O(n), S(n) ∈ O(n) for n reminders
    """

    __slots__ = ("_reminders",)

    def __init__(self) -> None:
        """Initialize ReminderEngine.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._reminders: Dict[str, Reminder] = {}

    def add(self, message: str, remind_at: str) -> Reminder:
        """Add a new reminder.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        reminder_id = uuid.uuid4().hex[:16]
        reminder = Reminder(id=reminder_id, message=message, remind_at=remind_at)
        object.__setattr__(self, "_reminders", {**self._reminders, reminder_id: reminder})
        return reminder

    def list(self) -> List[Reminder]:
        """List all reminders.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n reminders.
        """
        return list(self._reminders.values())

    def delete(self, reminder_id: str) -> bool:
        """Delete a reminder.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if reminder_id in self._reminders:
            object.__setattr__(self, "_reminders", {k: v for k, v in self._reminders.items() if k != reminder_id})
            return True
        return False


@dataclass(frozen=True, slots=True)
class Note:
    """
    Note data structure.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    id: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class NoteStore:
    """
    Note storage for Personal Assistant.

    Complexity:
        add: T(n) ∈ O(1), S(n) ∈ O(1)
        list: T(n) ∈ O(n), S(n) ∈ O(n) for n notes
    """

    __slots__ = ("_notes",)

    def __init__(self) -> None:
        """Initialize NoteStore.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._notes: Dict[str, Note] = {}

    def add(self, content: str) -> Note:
        """Add a new note.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        note_id = uuid.uuid4().hex[:16]
        note = Note(id=note_id, content=content)
        object.__setattr__(self, "_notes", {**self._notes, note_id: note})
        return note

    def list(self) -> List[Note]:
        """List all notes.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n notes.
        """
        return list(self._notes.values())

    def delete(self, note_id: str) -> bool:
        """Delete a note.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if note_id in self._notes:
            object.__setattr__(self, "_notes", {k: v for k, v in self._notes.items() if k != note_id})
            return True
        return False


@dataclass(frozen=True, slots=True)
class PlanStep:
    """
    Plan step data structure.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    step_number: int
    description: str
    completed: bool = False


@dataclass(frozen=True, slots=True)
class Plan:
    """
    Plan data structure.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    id: str
    title: str
    steps: Tuple[PlanStep, ...]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class QuickPlan:
    """
    Quick plan generator for Personal Assistant.

    Complexity:
        create: T(n) ∈ O(n), S(n) ∈ O(n) for n steps
    """

    __slots__ = ("_plans",)

    def __init__(self) -> None:
        """Initialize QuickPlan.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._plans: Dict[str, Plan] = {}

    def create(self, title: str, steps: List[str]) -> Plan:
        """Create a new plan.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n steps.
        """
        plan_id = uuid.uuid4().hex[:16]
        plan_steps = tuple(
            PlanStep(step_number=i, description=step)
            for i, step in enumerate(steps, 1)
        )
        plan = Plan(id=plan_id, title=title, steps=plan_steps)
        object.__setattr__(self, "_plans", {**self._plans, plan_id: plan})
        return plan

    def list(self) -> List[Plan]:
        """List all plans.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n plans.
        """
        return list(self._plans.values())


class ToneController:
    """
    Tone controller for Personal Assistant.

    Complexity:
        apply: T(n) ∈ O(n), S(n) ∈ O(n) for n text length
    """

    __slots__ = ("_tone_profile",)

    def __init__(self, tone_profile: str = "professional") -> None:
        """Initialize ToneController.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._tone_profile = tone_profile

    def apply(self, text: str) -> str:
        """Apply tone to text.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n text length.
        """
        if self._tone_profile == "professional":
            return self._apply_professional(text)
        elif self._tone_profile == "casual":
            return self._apply_casual(text)
        elif self._tone_profile == "formal":
            return self._apply_formal(text)
        return text

    def _apply_professional(self, text: str) -> str:
        """Apply professional tone.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n text length.
        """
        # Add polite prefixes/suffixes
        if not text.startswith(("I", "The", "Your", "Please")):
            text = "I " + text[0].lower() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        return text

    def _apply_casual(self, text: str) -> str:
        """Apply casual tone.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n text length.
        """
        # More relaxed language
        text = text.replace("I will", "I'll")
        text = text.replace("you are", "you're")
        text = text.replace("cannot", "can't")
        return text

    def _apply_formal(self, text: str) -> str:
        """Apply formal tone.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n text length.
        """
        # More formal language
        text = text.replace("I'll", "I will")
        text = text.replace("you're", "you are")
        text = text.replace("can't", "cannot")
        return text


class InfoLookup:
    """
    Info lookup tool for quick answers.

    Uses Main AI's knowledge router for answers.

    Complexity:
        lookup: T(n) ∈ O(k) for k knowledge items
    """

    __slots__ = ("_brain_controller",)

    def __init__(self, brain_controller: "BrainController") -> None:
        """Initialize InfoLookup.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._brain_controller = brain_controller

    def lookup(self, query: str) -> str:
        """Look up information using Main AI's knowledge router.

        Complexity: T(n) ∈ O(k) for k knowledge items.
        """
        try:
            knowledge_items = self._brain_controller.knowledge_router.route(
                query=query,
                domain="general",
                top_k=3,
            )
            if knowledge_items:
                return knowledge_items[0].content
            return "I don't have information about that."
        except Exception:
            return "I encountered an error looking up that information."
