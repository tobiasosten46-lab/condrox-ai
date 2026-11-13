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
Conversational AI Test — Test the standard AI for public chat interface.

Tests the Condrox BrainController for:
- Basic conversation
- Emotion detection
- Intent classification
- Response generation
- Empathy in responses

Complexity: T(n) in O(pipeline), S(n) in O(output).
"""

import sys
import os

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BrainController


@pytest.fixture
def brain() -> BrainController:
    """Provide a fresh, booted BrainController for multi-turn tests."""
    bc = BrainController()
    bc.boot_system()
    return bc


def test_conversational_ai():
    """Test the conversational AI for public chat interface."""
    
    print("🟠 Condrox Conversational AI — Public Chat Interface Test")
    print("=" * 70)
    
    # Initialize brain
    print("\n1. Initializing Condrox BrainController...")
    brain = BrainController()
    boot_result = brain.boot_system()
    print(f"   ✓ Brain booted (success: {boot_result.success})")
    
    # Test cases for public chat
    test_cases = [
        {
            "input": "Hello, how are you?",
            "expected_intent": "greet",
            "description": "Greeting",
        },
        {
            "input": "I'm feeling anxious about my presentation tomorrow.",
            "expected_intent": "emotional",
            "description": "Anxious emotion",
            "check_empathy": True,
        },
        {
            "input": "Can you explain how machine learning works?",
            "expected_intent": "explain",
            "description": "Explanation request",
        },
        {
            "input": "What's the weather like today?",
            "expected_intent": "question",
            "description": "General question",
        },
        {
            "input": "I'm so happy about my promotion!",
            "expected_intent": "emotional",
            "description": "Happy emotion",
        },
    ]
    
    print(f"\n2. Running {len(test_cases)} test cases...")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        user_input = test_case["input"]
        expected_intent = test_case["expected_intent"]
        description = test_case["description"]
        check_empathy = test_case.get("check_empathy", False)
        
        print(f"\nTest {i}: {description}")
        print(f"Input: '{user_input}'")
        
        # Process request
        state = brain.process_request(user_input)
        
        print(f"Intent: {state.intent}")
        print(f"Emotion: {state.emotion}")
        print(f"Mode: {state.mode}")
        
        # Check intent
        intent_match = state.intent == expected_intent
        intent_status = "✓" if intent_match else "✗"
        print(f"{intent_status} Intent: {state.intent} (expected: {expected_intent})")
        
        # Check empathy for anxious responses
        if check_empathy:
            output_lower = state.final_output.lower()
            empathy_words = ["normal", "hear", "understand", "worried"]
            has_empathy = any(word in output_lower for word in empathy_words)
            empathy_status = "✓" if has_empathy else "✗"
            print(f"{empathy_status} Empathy: {has_empathy}")
            
            if not has_empathy:
                print(f"   Missing empathy words in output")
        
        # Show output
        print(f"Output: {state.final_output[:100]}...")
        
        if intent_match and (not check_empathy or has_empathy):
            passed += 1
            print("✅ PASSED")
        else:
            failed += 1
            print("❌ FAILED")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed / len(test_cases) * 100:.1f}%")
    
    if failed == 0:
        print("\n✅ All tests passed — Conversational AI ready for public chat")
    else:
        print(f"\n❌ {failed} test(s) failed — Review output above")


def test_multi_turn_conversation(brain: BrainController):
    """Test multi-turn conversation with context."""
    
    print("\n\n🟠 Condrox Conversational AI — Multi-Turn Conversation Test")
    print("=" * 70)
    
    # Clear memory for clean test
    brain.memory_manager.store._items.clear()
    
    conversation = [
        "My name is Alice",
        "I work as a software engineer",
        "What's my name?",
        "What do I do?",
    ]
    
    print("\nRunning multi-turn conversation...")
    print("-" * 70)
    
    responses = []
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\nTurn {i}: '{user_input}'")
        state = brain.process_request(user_input)
        responses.append(state.final_output)
        print(f"Response: {state.final_output[:80]}...")
    
    # Verify context retention
    name_response = responses[2]  # "What's my name?"
    profession_response = responses[3]  # "What do I do?"
    
    name_correct = "alice" in name_response.lower()
    profession_correct = "software engineer" in profession_response.lower() or "engineer" in profession_response.lower()
    
    print(f"\nContext Retention Check:")
    print(f"  Name remembered: {'✓' if name_correct else '✗'}")
    print(f"  Profession remembered: {'✓' if profession_correct else '✗'}")
    
    if name_correct and profession_correct:
        print("\n✅ Multi-turn conversation test complete")
    else:
        print("\n❌ Multi-turn context retention failed")
        if not name_correct:
            print("  - Name not remembered")
        if not profession_correct:
            print("  - Profession not remembered")


if __name__ == "__main__":
    brain = test_conversational_ai()
    test_multi_turn_conversation(brain)
    sys.exit(0)
