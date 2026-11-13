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
Workflow Layer — Behavioral Logic for Personal Assistant Agent.

Creates workflows that make the agent useful in practice:
- "Create task" → save → confirm → offer follow-up
- "Remind me" → parse time → save → confirm
- "Plan X" → generate steps → save → offer follow-up
- "Take note" → save → confirm → offer display
- "What's my tasks?" → retrieve → present structured

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

from agents.PersonalAgent.src.capability_layer import (
    InfoLookup,
    NoteStore,
    Plan,
    QuickPlan,
    ReminderEngine,
    TaskManager,
    TaskPriority,
    ToneController,
)
from agents.PersonalAgent.src.role_definition import RoleDefinition, ToneProfile


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """
    Result from workflow execution.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    success: bool
    message: str
    follow_up: str = ""
    data: Optional[dict] = None


class WorkflowLayer:
    """
    Workflow orchestration for Personal Assistant.

    Combines capability tools into useful behavioral patterns.

    Complexity:
        execute: T(n) ∈ O(tool) for specific tool operation
    """

    __slots__ = (
        "_brain_controller",
        "_role_definition",
        "_task_manager",
        "_reminder_engine",
        "_note_store",
        "_quick_plan",
        "_info_lookup",
        "_tone_controller",
    )

    def __init__(
        self,
        brain_controller: "BrainController",
        role_definition: RoleDefinition,
    ) -> None:
        """Initialize WorkflowLayer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._brain_controller = brain_controller
        self._role_definition = role_definition
        self._task_manager = TaskManager()
        self._reminder_engine = ReminderEngine()
        self._note_store = NoteStore()
        self._quick_plan = QuickPlan()
        self._info_lookup = InfoLookup(brain_controller)
        self._tone_controller = ToneController(tone_profile=role_definition.tone_profile.value)

    def execute(self, user_input: str) -> WorkflowResult:
        """Execute appropriate workflow based on user input.

        Complexity: T(n) ∈ O(pattern) for pattern matching + tool operation.
        """
        lower_input = user_input.lower()

        # "Create task" workflow
        if self._detect_create_task(lower_input):
            return self._workflow_create_task(user_input)

        # "Remind me" workflow
        if self._detect_remind_me(lower_input):
            return self._workflow_remind_me(user_input)

        # "Plan X" workflow
        if self._detect_plan(lower_input):
            return self._workflow_plan(user_input)

        # "Take note" workflow
        if self._detect_take_note(lower_input):
            return self._workflow_take_note(user_input)

        # "What's my tasks?" workflow
        if self._detect_list_tasks(lower_input):
            return self._workflow_list_tasks()

        # "What's my notes?" workflow
        if self._detect_list_notes(lower_input):
            return self._workflow_list_notes()

        # Info lookup workflow
        if self._detect_question(lower_input):
            return self._workflow_info_lookup(user_input)

        # Default: no specific workflow
        return WorkflowResult(
            success=False,
            message="I didn't recognize a specific command. Try 'create task', 'remind me', 'plan', 'take note', or 'what are my tasks'.",
        )

    def _detect_create_task(self, text: str) -> bool:
        """Detect create task intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        patterns = [
            r"create task",
            r"add task",
            r"new task",
            r"todo",
            r"to do",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _detect_remind_me(self, text: str) -> bool:
        """Detect remind me intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        patterns = [
            r"remind me",
            r"reminder",
            r"remind",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _detect_plan(self, text: str) -> bool:
        """Detect plan intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        patterns = [
            r"plan for",
            r"create plan",
            r"make a plan",
            r"planning",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _detect_take_note(self, text: str) -> bool:
        """Detect take note intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        patterns = [
            r"take note",
            r"note down",
            r"save note",
            r"remember this",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _detect_list_tasks(self, text: str) -> bool:
        """Detect list tasks intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        patterns = [
            r"what.?s my task",
            r"list task",
            r"show task",
            r"my task",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _detect_list_notes(self, text: str) -> bool:
        """Detect list notes intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        patterns = [
            r"what.?s my note",
            r"list note",
            r"show note",
            r"my note",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _detect_question(self, text: str) -> bool:
        """Detect question intent.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        return text.endswith("?") or text.startswith(("what", "how", "why", "when", "where", "who"))

    def _workflow_create_task(self, user_input: str) -> WorkflowResult:
        """Execute create task workflow: save → confirm → offer follow-up.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Extract task title from input
        task_title = self._extract_task_title(user_input)
        if not task_title:
            return WorkflowResult(
                success=False,
                message="I couldn't extract a task title. Please specify what you want to do.",
            )

        # Create task
        task = self._task_manager.add(title=task_title)

        # Generate confirmation with tone
        confirmation = self._tone_controller.apply(f"Task created: {task.title}")
        follow_up = self._tone_controller.apply("Would you like me to set a reminder for this task?")

        return WorkflowResult(
            success=True,
            message=confirmation,
            follow_up=follow_up,
            data={"task_id": task.id, "task_title": task.title},
        )

    def _workflow_remind_me(self, user_input: str) -> bool:
        """Execute remind me workflow: parse time → save → confirm.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Extract reminder message and time
        reminder_data = self._extract_reminder_data(user_input)
        if not reminder_data:
            return WorkflowResult(
                success=False,
                message="I couldn't parse the reminder. Please specify what to remind you and when.",
            )

        message, remind_at = reminder_data

        # Create reminder
        reminder = self._reminder_engine.add(message=message, remind_at=remind_at)

        # Generate confirmation with tone
        confirmation = self._tone_controller.apply(f"Reminder set: {reminder.message} at {reminder.remind_at}")

        return WorkflowResult(
            success=True,
            message=confirmation,
            data={"reminder_id": reminder.id, "message": reminder.message, "remind_at": reminder.remind_at},
        )

    def _workflow_plan(self, user_input: str) -> WorkflowResult:
        """Execute plan workflow: generate steps → save → offer follow-up.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n steps.
        """
        # Extract plan topic
        plan_topic = self._extract_plan_topic(user_input)
        if not plan_topic:
            return WorkflowResult(
                success=False,
                message="I couldn't extract a plan topic. Please specify what you want to plan.",
            )

        # Generate simple steps (placeholder - would use Main AI for real generation)
        steps = [
            f"Research {plan_topic}",
            f"Define goals for {plan_topic}",
            f"Create timeline for {plan_topic}",
            f"Execute plan for {plan_topic}",
            f"Review and adjust {plan_topic}",
        ]

        # Create plan
        plan = self._quick_plan.create(title=plan_topic, steps=steps)

        # Generate confirmation with tone
        confirmation = self._tone_controller.apply(f"Plan created: {plan.title} with {len(plan.steps)} steps")
        follow_up = self._tone_controller.apply("Would you like me to add this to your tasks?")

        return WorkflowResult(
            success=True,
            message=confirmation,
            follow_up=follow_up,
            data={"plan_id": plan.id, "plan_title": plan.title, "steps": [s.description for s in plan.steps]},
        )

    def _workflow_take_note(self, user_input: str) -> WorkflowResult:
        """Execute take note workflow: save → confirm → offer display.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Extract note content
        note_content = self._extract_note_content(user_input)
        if not note_content:
            return WorkflowResult(
                success=False,
                message="I couldn't extract note content. Please specify what you want me to remember.",
            )

        # Save note
        note = self._note_store.add(content=note_content)

        # Generate confirmation with tone
        confirmation = self._tone_controller.apply(f"Note saved: {note.content[:50]}...")
        follow_up = self._tone_controller.apply("Would you like me to show your notes?")

        return WorkflowResult(
            success=True,
            message=confirmation,
            follow_up=follow_up,
            data={"note_id": note.id, "note_content": note.content},
        )

    def _workflow_list_tasks(self) -> WorkflowResult:
        """Execute list tasks workflow: retrieve → present structured.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n tasks.
        """
        tasks = self._task_manager.list(completed=False)

        if not tasks:
            return WorkflowResult(
                success=True,
                message="You have no pending tasks.",
            )

        # Format tasks
        task_list = "\n".join(
            f"- {task.title} (Priority: {task.priority.value})"
            for task in tasks
        )

        # Generate response with tone
        response = self._tone_controller.apply(f"Your pending tasks:\n{task_list}")

        return WorkflowResult(
            success=True,
            message=response,
            data={"task_count": len(tasks), "tasks": [{"title": t.title, "priority": t.priority.value} for t in tasks]},
        )

    def _workflow_list_notes(self) -> WorkflowResult:
        """Execute list notes workflow: retrieve → present structured.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n) for n notes.
        """
        notes = self._note_store.list()

        if not notes:
            return WorkflowResult(
                success=True,
                message="You have no saved notes.",
            )

        # Format notes
        note_list = "\n".join(f"- {note.content}" for note in notes)

        # Generate response with tone
        response = self._tone_controller.apply(f"Your notes:\n{note_list}")

        return WorkflowResult(
            success=True,
            message=response,
            data={"note_count": len(notes), "notes": [{"content": n.content} for n in notes]},
        )

    def _workflow_info_lookup(self, user_input: str) -> WorkflowResult:
        """Execute info lookup workflow: use Main AI knowledge router.

        Complexity: T(n) ∈ O(k) for k knowledge items.
        """
        answer = self._info_lookup.lookup(user_input)

        # Apply tone
        response = self._tone_controller.apply(answer)

        return WorkflowResult(
            success=True,
            message=response,
        )

    def _extract_task_title(self, user_input: str) -> str:
        """Extract task title from user input.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Remove command words
        patterns = [
            r"create task\s*[:to]?\s*",
            r"add task\s*[:to]?\s*",
            r"new task\s*[:to]?\s*",
            r"todo\s*[:to]?\s*",
            r"to do\s*[:to]?\s*",
        ]
        for pattern in patterns:
            user_input = re.sub(pattern, "", user_input, flags=re.IGNORECASE)
        return user_input.strip()

    def _extract_reminder_data(self, user_input: str) -> Optional[tuple[str, str]]:
        """Extract reminder message and time from user input.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Simple extraction - would be more sophisticated in production
        # Format: "remind me to [message] at [time]"
        match = re.search(r"remind me to (.+?) at (.+)", user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None

    def _extract_plan_topic(self, user_input: str) -> str:
        """Extract plan topic from user input.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Remove command words
        patterns = [
            r"plan for\s*",
            r"create plan for\s*",
            r"make a plan for\s*",
            r"planning\s*",
        ]
        for pattern in patterns:
            user_input = re.sub(pattern, "", user_input, flags=re.IGNORECASE)
        return user_input.strip()

    def _extract_note_content(self, user_input: str) -> str:
        """Extract note content from user input.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Remove command words
        patterns = [
            r"take note\s*[:to]?\s*",
            r"note down\s*[:to]?\s*",
            r"save note\s*[:to]?\s*",
            r"remember this\s*[:to]?\s*",
        ]
        for pattern in patterns:
            user_input = re.sub(pattern, "", user_input, flags=re.IGNORECASE)
        return user_input.strip()
