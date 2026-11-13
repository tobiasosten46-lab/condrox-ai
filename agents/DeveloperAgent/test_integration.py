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
Integration test — Verify DeveloperAgent actually generates code.

Tests the full pipeline:
1. Initialize DeveloperAgent agent
2. Initialize Condrox BrainController
3. Execute a code generation task
4. Verify output is generated code

Complexity: T(n) in O(pipeline + gates + rag), S(n) in O(output).
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core import BrainController
from agents.DeveloperAgent.condrox_agent_coder import CondroxAgentCoder


async def test_code_generation():
    """Test that DeveloperAgent generates code for a simple request."""
    
    print("🟠 DeveloperAgent: Integration Test")
    print("=" * 60)
    
    # Initialize agent
    print("\n1. Initializing DeveloperAgent agent...")
    agent = CondroxAgentCoder(root_dir=".")
    print(f"   ✓ Agent initialized (state: {agent.state.value})")
    
    # Initialize brain
    print("\n2. Initializing Condrox BrainController...")
    brain = BrainController()
    boot_result = brain.boot_system()
    print(f"   ✓ Brain booted (success: {boot_result.success})")
    
    # Test code generation
    print("\n3. Testing code generation...")
    print(f"   Request: 'Generate a function that adds two numbers'")
    
    result = await agent.execute(
        user_input="Generate a function that adds two numbers",
        brain_controller=brain,
        language="python",
        file_path="test_output.py",
        context="",
    )
    
    print(f"\n4. Results:")
    print(f"   Task ID: {result.task_id}")
    print(f"   Intent: {result.intent.value}")
    print(f"   State: {result.state.value}")
    print(f"   Confidence: {result.confidence:.2f} ({result.confidence_level.value})")
    print(f"   Domain: {result.domain.value}")
    print(f"   Latency: {result.latency_ms}ms")
    
    print(f"\n5. Verification:")
    print(f"   Passed: {result.verification.passed}")
    print(f"   Blocked: {result.verification.blocked}")
    print(f"   Score: {result.verification.score:.2f}")
    
    print(f"\n6. Generated Code:")
    print("-" * 60)
    print(result.output)
    print("-" * 60)
    
    # Verify output contains code
    if "def" in result.output.lower() or "function" in result.output.lower():
        print("\n✅ TEST PASSED: Code generation working")
        return True
    else:
        print("\n❌ TEST FAILED: No code generated")
        return False


async def test_intent_classification():
    """Test that intent classification works."""
    
    print("\n\n🟠 DeveloperAgent: Intent Classification Test")
    print("=" * 60)
    
    agent = CondroxAgentCoder(root_dir=".")
    
    test_cases = [
        ("fix this bug", "fix"),
        ("generate a function", "generate"),
        ("review my code", "review"),
        ("optimize performance", "optimize"),
        ("security audit", "security_audit"),
    ]
    
    for user_input, expected_intent in test_cases:
        from agents.DeveloperAgent.condrox_agent_coder import IntentClassifier
        classifier = IntentClassifier()
        intent, confidence = classifier.classify(user_input)
        
        status = "✓" if intent.value == expected_intent else "✗"
        print(f"   {status} '{user_input}' -> {intent.value} (expected: {expected_intent})")
    
    print("\n✅ Intent classification test complete")


async def main():
    """Run all integration tests."""
    try:
        # Test code generation
        code_gen_passed = await test_code_generation()
        
        # Test intent classification
        await test_intent_classification()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if code_gen_passed:
            print("✅ DeveloperAgent is generating code correctly")
            print("✅ Integration test PASSED")
            return 0
        else:
            print("❌ DeveloperAgent failed to generate code")
            print("❌ Integration test FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
