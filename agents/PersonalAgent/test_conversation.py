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
Advanced Conversation Test for PersonalAgent

Comprehensive test suite that maps out all functionality and identifies issues.

Tests:
- PersonalAgent initialization with BrainController
- Conversation processing through Main AI
- Context management (get/clear)
- Error handling
- All API endpoints
- Integration with Main AI pipeline

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.brain.brain_controller import BrainController
from core.brain.brain_config import BrainConfig
from agents.PersonalAgent.personal_agent import PersonalAgent, PersonalAgentState


class ConversationTest:
    """Advanced conversation test suite for PersonalAgent."""

    def __init__(self):
        self.brain = None
        self.agent = None
        self.test_results = []

    def log(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"  {message}")

    def test_01_brain_initialization(self):
        """Test BrainController initialization."""
        try:
            config = BrainConfig()
            self.brain = BrainController(config)
            self.log("BrainController initialization", True)
        except Exception as e:
            self.log("BrainController initialization", False, str(e))

    def test_02_brain_boot(self):
        """Test BrainController boot system."""
        if self.brain is None:
            self.log("BrainController boot", False, "Brain not initialized")
            return

        try:
            report = self.brain.boot_system()
            self.log("BrainController boot", report.success, f"Success: {report.success}")
        except Exception as e:
            self.log("BrainController boot", False, str(e))

    def test_03_agent_initialization(self):
        """Test PersonalAgent initialization with BrainController."""
        if self.brain is None:
            self.log("PersonalAgent initialization", False, "Brain not initialized")
            return

        try:
            self.agent = PersonalAgent(self.brain)
            self.log("PersonalAgent initialization", True)
        except Exception as e:
            self.log("PersonalAgent initialization", False, str(e))

    def test_04_agent_state(self):
        """Test agent initial state."""
        if self.agent is None:
            self.log("Agent initial state", False, "Agent not initialized")
            return

        try:
            state = self.agent.state
            status = state == PersonalAgentState.IDLE
            self.log("Agent initial state", status, f"State: {state.value}")
        except Exception as e:
            self.log("Agent initial state", False, str(e))

    def test_05_conversation_greeting(self):
        """Test basic greeting conversation."""
        if self.agent is None:
            self.log("Conversation greeting", False, "Agent not initialized")
            return

        try:
            result = self.agent.process("Hello, how are you?")
            has_response = bool(result.response)
            self.log("Conversation greeting", has_response, f"Response: {result.response[:100]}")
        except Exception as e:
            self.log("Conversation greeting", False, str(e))

    def test_06_conversation_question(self):
        """Test question conversation."""
        if self.agent is None:
            self.log("Conversation question", False, "Agent not initialized")
            return

        try:
            result = self.agent.process("What is Python?")
            has_response = bool(result.response)
            self.log("Conversation question", has_response, f"Response: {result.response[:100]}")
        except Exception as e:
            self.log("Conversation question", False, str(e))

    def test_07_conversation_empty_input(self):
        """Test empty input handling."""
        if self.agent is None:
            self.log("Empty input handling", False, "Agent not initialized")
            return

        try:
            result = self.agent.process("")
            has_response = bool(result.response)
            self.log("Empty input handling", has_response, f"Response: {result.response[:100]}")
        except Exception as e:
            self.log("Empty input handling", False, str(e))

    def test_08_conversation_context(self):
        """Test conversation context retrieval."""
        if self.agent is None:
            self.log("Conversation context", False, "Agent not initialized")
            return

        try:
            context = self.agent.get_conversation_context()
            has_context = isinstance(context, dict)
            self.log("Conversation context", has_context, f"Context keys: {list(context.keys())}")
        except Exception as e:
            self.log("Conversation context", False, str(e))

    def test_09_conversation_clear_context(self):
        """Test clearing conversation context."""
        if self.agent is None:
            self.log("Clear conversation context", False, "Agent not initialized")
            return

        try:
            self.agent.clear_conversation_context()
            self.log("Clear conversation context", True)
        except Exception as e:
            self.log("Clear conversation context", False, str(e))

    def test_10_agent_status(self):
        """Test agent status method."""
        if self.agent is None:
            self.log("Agent status", False, "Agent not initialized")
            return

        try:
            status = self.agent.status()
            has_status = isinstance(status, dict)
            self.log("Agent status", has_status, f"Status: {status}")
        except Exception as e:
            self.log("Agent status", False, str(e))

    def test_11_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        if self.agent is None:
            self.log("Multi-turn conversation", False, "Agent not initialized")
            return

        try:
            # First turn
            result1 = self.agent.process("My name is Alice")
            # Second turn
            result2 = self.agent.process("What is my name?")
            has_responses = bool(result1.response) and bool(result2.response)
            self.log("Multi-turn conversation", has_responses, f"Turn 1: {result1.response[:50]}, Turn 2: {result2.response[:50]}")
        except Exception as e:
            self.log("Multi-turn conversation", False, str(e))

    def test_12_brain_tools_available(self):
        """Test that BrainController has universal tools."""
        if self.brain is None:
            self.log("Brain tools available", False, "Brain not initialized")
            return

        try:
            has_file_ops = hasattr(self.brain, 'file_ops')
            has_shell_exec = hasattr(self.brain, 'shell_exec')
            has_git_ops = hasattr(self.brain, 'git_ops')
            all_tools = has_file_ops and has_shell_exec and has_git_ops
            self.log("Brain tools available", all_tools, f"FileOps: {has_file_ops}, ShellExec: {has_shell_exec}, GitOps: {has_git_ops}")
        except Exception as e:
            self.log("Brain tools available", False, str(e))

    def test_13_brain_ci_components(self):
        """Test that BrainController has CI components."""
        if self.brain is None:
            self.log("Brain CI components", False, "Brain not initialized")
            return

        try:
            has_memory_graph = hasattr(self.brain, 'memory_graph')
            has_multi_turn = hasattr(self.brain, 'multi_turn_reasoning')
            has_knowledge_router = hasattr(self.brain, 'knowledge_router')
            all_ci = has_memory_graph and has_multi_turn and has_knowledge_router
            self.log("Brain CI components", all_ci, f"MemoryGraph: {has_memory_graph}, MultiTurn: {has_multi_turn}, KnowledgeRouter: {has_knowledge_router}")
        except Exception as e:
            self.log("Brain CI components", False, str(e))

    def test_14_pipeline_stages(self):
        """Test that BrainController has pipeline stages."""
        if self.brain is None:
            self.log("Pipeline stages", False, "Brain not initialized")
            return

        try:
            stage_count = self.brain.pipeline.stage_count()
            has_stages = stage_count > 0
            self.log("Pipeline stages", has_stages, f"Stage count: {stage_count}")
        except Exception as e:
            self.log("Pipeline stages", False, str(e))

    def test_15_agent_request_count(self):
        """Test agent request count tracking."""
        if self.agent is None:
            self.log("Agent request count", False, "Agent not initialized")
            return

        try:
            initial_count = self.agent.request_count
            self.agent.process("Test message")
            final_count = self.agent.request_count
            count_increased = final_count > initial_count
            self.log("Agent request count", count_increased, f"Initial: {initial_count}, Final: {final_count}")
        except Exception as e:
            self.log("Agent request count", False, str(e))

    def test_16_conversation_with_context(self):
        """Test conversation with additional context."""
        if self.agent is None:
            self.log("Conversation with context", False, "Agent not initialized")
            return

        try:
            result = self.agent.process("Help me with Python", context="I'm a beginner")
            has_response = bool(result.response)
            self.log("Conversation with context", has_response, f"Response: {result.response[:100]}")
        except Exception as e:
            self.log("Conversation with context", False, str(e))

    def test_17_brain_status(self):
        """Test BrainController status method."""
        if self.brain is None:
            self.log("Brain status", False, "Brain not initialized")
            return

        try:
            status = self.brain.status()
            has_status = isinstance(status, dict)
            self.log("Brain status", has_status, f"Status keys: {list(status.keys())[:10]}")
        except Exception as e:
            self.log("Brain status", False, str(e))

    def test_18_agent_uses_brain(self):
        """Test that agent actually uses BrainController."""
        if self.agent is None or self.brain is None:
            self.log("Agent uses Brain", False, "Agent or Brain not initialized")
            return

        try:
            # Check that agent has reference to brain
            uses_brain = self.agent._brain_controller is self.brain
            self.log("Agent uses Brain", uses_brain, f"Same instance: {uses_brain}")
        except Exception as e:
            self.log("Agent uses Brain", False, str(e))

    def test_19_conversation_emotion(self):
        """Test conversation emotion detection."""
        if self.agent is None:
            self.log("Conversation emotion", False, "Agent not initialized")
            return

        try:
            result = self.agent.process("I'm very happy today!")
            has_emotion = bool(result.emotion)
            self.log("Conversation emotion", has_emotion, f"Emotion: {result.emotion}")
        except Exception as e:
            self.log("Conversation emotion", False, str(e))

    def test_20_conversation_confidence(self):
        """Test conversation confidence score."""
        if self.agent is None:
            self.log("Conversation confidence", False, "Agent not initialized")
            return

        try:
            result = self.agent.process("What is AI?")
            has_confidence = 0.0 <= result.confidence <= 1.0
            self.log("Conversation confidence", has_confidence, f"Confidence: {result.confidence}")
        except Exception as e:
            self.log("Conversation confidence", False, str(e))

    def run_all_tests(self):
        """Run all tests."""
        print("=" * 60)
        print("PersonalAgent Advanced Conversation Test Suite")
        print("=" * 60)
        print()

        # BrainController tests
        self.test_01_brain_initialization()
        self.test_02_brain_boot()
        self.test_12_brain_tools_available()
        self.test_13_brain_ci_components()
        self.test_14_pipeline_stages()
        self.test_17_brain_status()

        # PersonalAgent tests
        self.test_03_agent_initialization()
        self.test_04_agent_state()
        self.test_05_conversation_greeting()
        self.test_06_conversation_question()
        self.test_07_conversation_empty_input()
        self.test_08_conversation_context()
        self.test_09_conversation_clear_context()
        self.test_10_agent_status()
        self.test_11_multi_turn_conversation()
        self.test_15_agent_request_count()
        self.test_16_conversation_with_context()
        self.test_18_agent_uses_brain()
        self.test_19_conversation_emotion()
        self.test_20_conversation_confidence()

        # Summary
        print()
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")

        if total - passed > 0:
            print()
            print("Failed tests:")
            for name, passed, message in self.test_results:
                if not passed:
                    print(f"  - {name}: {message}")

        # Cleanup
        if self.brain is not None:
            try:
                self.brain.close()
            except Exception:
                pass

        return passed == total


def main():
    """Main entry point."""
    test = ConversationTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
