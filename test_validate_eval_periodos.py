# Test Script for validate_eval_periodos.py

import json
import sys
from validate_eval_periodos import validate_and_adjust_n_eval_periodos


def test_case(name, model_periodo, n_eval_periodos, ult_periodo_data, expected_adjusted):
    """Run a test case and display results."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Input:")
    print(f"  model_periodo: {model_periodo}")
    print(f"  n_eval_periodos: {n_eval_periodos}")
    print(f"  ult_periodo_data: {ult_periodo_data}")
    
    adjusted, max_available, was_adjusted = validate_and_adjust_n_eval_periodos(
        model_periodo, n_eval_periodos, ult_periodo_data
    )
    
    print(f"\nOutput:")
    print(f"  Max available periods: {max_available}")
    print(f"  Adjusted n_eval_periodos: {adjusted}")
    print(f"  Was adjusted: {was_adjusted}")
    
    # Verify expected result
    if adjusted == expected_adjusted:
        print(f"  ✅ TEST PASSED (expected {expected_adjusted})")
    else:
        print(f"  ❌ TEST FAILED (expected {expected_adjusted}, got {adjusted})")
        sys.exit(1)


def main():
    """Run all test cases."""
    print("Running validate_eval_periodos.py tests...")
    
    # Test 1: Your example - needs adjustment
    test_case(
        name="Requested 10 periods but only 7 available",
        model_periodo=202506,
        n_eval_periodos=10,
        ult_periodo_data=202512,
        expected_adjusted=7
    )
    
    # Test 2: Valid request within range
    test_case(
        name="Requested 5 periods with 7 available",
        model_periodo=202506,
        n_eval_periodos=5,
        ult_periodo_data=202512,
        expected_adjusted=5
    )
    
    # Test 3: Exact match
    test_case(
        name="Requested exactly max available (7)",
        model_periodo=202506,
        n_eval_periodos=7,
        ult_periodo_data=202512,
        expected_adjusted=7
    )
    
    # Test 4: Cross year boundary
    test_case(
        name="Cross year boundary (Dec 2025 to Mar 2026)",
        model_periodo=202512,
        n_eval_periodos=10,
        ult_periodo_data=202603,
        expected_adjusted=4
    )
    
    # Test 5: Only 1 period available
    test_case(
        name="Only 1 period available",
        model_periodo=202512,
        n_eval_periodos=5,
        ult_periodo_data=202512,
        expected_adjusted=1
    )
    
    print(f"\n{'='*60}")
    print("✅ ALL TESTS PASSED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
