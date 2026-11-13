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
Extended Tools Layer — Advanced tools for Personal Assistant Agent.

Provides advanced tools that make the agent more useful:
- TimeParser (parse "i morgen", "kl 14", "neste uke")
- PriorityClassifier (high/medium/low priority)
- SmartSuggest (suggest follow-up)
- AutoPlan (generate plan based on user's goals)
- DailyBrief (daily summary of tasks and notes)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

from agents.PersonalAgent.src.capability_layer import TaskPriority


class TimeParser:
    """
    Time parser for natural language time expressions.

    Parses expressions like "i morgen", "kl 14", "neste uke", "i dag kl 10".

    Complexity:
        parse: T(n) ∈ O(n) for pattern matching, S(n) ∈ O(1)
    """

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize TimeParser.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        pass

    def parse(self, time_expression: str) -> Optional[str]:
        """Parse natural language time expression to ISO format.

        Args:
            time_expression: Natural language time expression.

        Returns:
            ISO format datetime string or None if parsing fails.

        Complexity: T(n) ∈ O(n) for pattern matching, S(n) ∈ O(1).
        """
        lower_expr = time_expression.lower()

        # "i morgen" / "tomorrow"
        if "i morgen" in lower_expr or "tomorrow" in lower_expr:
            tomorrow = datetime.now() + timedelta(days=1)
            # Extract time if present
            time_match = re.search(r"kl\s*(\d{1,2})(?::(\d{2}))?", lower_expr)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                tomorrow = tomorrow.replace(hour=hour, minute=minute)
            return tomorrow.isoformat()

        # "i dag" / "today"
        if "i dag" in lower_expr or "today" in lower_expr:
            today = datetime.now()
            time_match = re.search(r"kl\s*(\d{1,2})(?::(\d{2}))?", lower_expr)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                today = today.replace(hour=hour, minute=minute)
            return today.isoformat()

        # "neste uke" / "next week"
        if "neste uke" in lower_expr or "next week" in lower_expr:
            next_week = datetime.now() + timedelta(weeks=1)
            return next_week.isoformat()

        # "kl HH:MM" / "HH:MM"
        time_match = re.search(r"kl\s*(\d{1,2})(?::(\d{2}))?", lower_expr)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            today = datetime.now().replace(hour=hour, minute=minute)
            return today.isoformat()

        # "HH:MM"
        time_match = re.search(r"(\d{1,2}):(\d{2})", lower_expr)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            today = datetime.now().replace(hour=hour, minute=minute)
            return today.isoformat()

        # "om X dager" / "in X days"
        days_match = re.search(r"(?:om|in)\s*(\d+)\s*dag", lower_expr)
        if days_match:
            days = int(days_match.group(1))
            future = datetime.now() + timedelta(days=days)
            return future.isoformat()

        return None


class PriorityClassifier:
    """
    Priority classifier for tasks.

    Classifies task priority based on keywords and context.

    Complexity:
        classify: T(n) ∈ O(n) for keyword matching, S(n) ∈ O(1)
    """

    __slots__ = ("_high_keywords", "_low_keywords")

    def __init__(self) -> None:
        """Initialize PriorityClassifier.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._high_keywords = [
            "urgent", "critical", "important", "asap", "haster", "kritisk",
            "viktig", "nå", "nåværende", "emergency", "pressing",
        ]
        self._low_keywords = [
            "later", "someday", "eventually", "senere", "en dag",
            "når som helst", "low priority", "lav prioritet",
        ]

    def classify(self, task_description: str) -> TaskPriority:
        """Classify task priority based on description.

        Args:
            task_description: Task description.

        Returns:
            TaskPriority enum value.

        Complexity: T(n) ∈ O(n) for keyword matching, S(n) ∈ O(1).
        """
        lower_desc = task_description.lower()

        # Check for high priority keywords
        for keyword in self._high_keywords:
            if keyword in lower_desc:
                return TaskPriority.HIGH

        # Check for low priority keywords
        for keyword in self._low_keywords:
            if keyword in lower_desc:
                return TaskPriority.LOW

        # Default to medium
        return TaskPriority.MEDIUM


class SmartSuggest:
    """
    Smart suggest for follow-up suggestions.

    Suggests appropriate follow-up actions based on context.

    Complexity:
        suggest: T(n) ∈ O(n) for pattern matching, S(n) ∈ O(1)
    """

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize SmartSuggest.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        pass

    def suggest(self, context: str, last_action: str) -> str:
        """Suggest follow-up action based on context.

        Args:
            context: Current context (task, reminder, note, etc.)
            last_action: Last action performed.

        Returns:
            Suggested follow-up action.

        Complexity: T(n) ∈ O(n) for pattern matching, S(n) ∈ O(1).
        """
        if context == "task":
            return "Would you like me to set a reminder for this task?"
        elif context == "reminder":
            return "Would you like me to add this to your tasks?"
        elif context == "note":
            return "Would you like me to show your other notes?"
        elif context == "plan":
            return "Would you like me to break this down into smaller tasks?"
        else:
            return "Is there anything else I can help you with?"


class AutoPlan:
    """
    Auto plan generator based on user's goals.

    Generates plans automatically based on user goals and context.

    Complexity:
        generate: T(n) ∈ O(n) for plan generation, S(n) ∈ O(n) for n steps
    """

    __slots__ = ("_brain_controller",)

    def __init__(self, brain_controller: "BrainController") -> None:
        """Initialize AutoPlan.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._brain_controller = brain_controller

    def generate(self, goal: str) -> List[str]:
        """Generate plan steps based on user goal.

        Args:
            goal: User's goal.

        Returns:
            List of plan steps.

        Complexity: T(n) ∈ O(n) for plan generation, S(n) ∈ O(n) for n steps.
        """
        # Use Main AI to generate plan steps
        # For now, return a generic plan structure
        # In production, this would use the reasoning engine

        generic_steps = [
            f"Define specific objectives for: {goal}",
            f"Research requirements for: {goal}",
            f"Create timeline for: {goal}",
            f"Identify resources needed for: {goal}",
            f"Execute first phase of: {goal}",
            f"Monitor progress on: {goal}",
            f"Review and adjust: {goal}",
        ]

        return generic_steps


class DailyBrief:
    """
    Daily brief generator for tasks and notes.

    Generates a daily summary of tasks and notes.

    Complexity:
        generate: T(n) ∈ O(n) for n tasks and notes, S(n) ∈ O(n)
    """

    __slots__ = ("_task_manager", "_note_store")

    def __init__(self, task_manager, note_store) -> None:
        """Initialize DailyBrief.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._task_manager = task_manager
        self._note_store = note_store

    def generate(self) -> str:
        """Generate daily brief summary.

        Returns:
            Formatted daily brief.

        Complexity: T(n) ∈ O(n) for n tasks and notes, S(n) ∈ O(n).
        """
        # Get pending tasks
        pending_tasks = self._task_manager.list(completed=False)
        
        # Get recent notes
        notes = self._note_store.list()
        
        # Build brief
        brief_parts = []
        
        brief_parts.append("=== Daily Brief ===")
        brief_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        brief_parts.append("")
        
        # Tasks section
        brief_parts.append(f"Pending Tasks ({len(pending_tasks)}):")
        if pending_tasks:
            for task in pending_tasks[:5]:  # Show max 5 tasks
                brief_parts.append(f"  - {task.title} ({task.priority.value})")
            if len(pending_tasks) > 5:
                brief_parts.append(f"  ... and {len(pending_tasks) - 5} more")
        else:
            brief_parts.append("  No pending tasks")
        
        brief_parts.append("")
        
        # Notes section
        brief_parts.append(f"Recent Notes ({len(notes)}):")
        if notes:
            for note in notes[:3]:  # Show max 3 notes
                brief_parts.append(f"  - {note.content[:50]}...")
            if len(notes) > 3:
                brief_parts.append(f"  ... and {len(notes) - 3} more")
        else:
            brief_parts.append("  No notes")
        
        return "\n".join(brief_parts)
