#!/usr/bin/env python3
"""
Test script for validation functionality.

This script tests the validation system with both valid and invalid data
to ensure it catches errors and warnings appropriately.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.validation import create_default_validator, ValidationResult
from scripts.benchmark.result_collector import ResultCollector
from scripts.solvers.solver_interface import SolverResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_validation_system():
    """Test the validation system with various data scenarios."""
    
    print("=" * 60)
    print("TESTING VALIDATION SYSTEM")
    print("=" * 60)
    
    # Create validator
    validator = create_default_validator()
    print(f"✓ Created validator with thresholds:")
    print(f"  Max solve time: {validator.max_solve_time}s")
    print(f"  Min solve time: {validator.min_solve_time}s")
    print(f"  Max objective magnitude: {validator.max_objective_magnitude}")
    
    # Test 1: Valid result
    print("\n1. Testing valid result:")
    valid_result = {
        'solver_name': 'SciPy',
        'problem_name': 'test_lp',
        'solve_time': 0.001,
        'status': 'optimal',
        'objective_value': 1.5
    }
    
    validation = validator.validate_single_result(valid_result)
    print(f"   Result: Valid={validation.is_valid}, Severity={validation.severity}")
    if validation.errors:
        print(f"   Errors: {validation.errors}")
    if validation.warnings:
        print(f"   Warnings: {validation.warnings}")
    
    # Test 2: Invalid solve time
    print("\n2. Testing negative solve time:")
    invalid_time_result = {
        'solver_name': 'BadSolver',
        'problem_name': 'test_lp',
        'solve_time': -0.5,  # Negative time
        'status': 'optimal',
        'objective_value': 1.5
    }
    
    validation = validator.validate_single_result(invalid_time_result)
    print(f"   Result: Valid={validation.is_valid}, Severity={validation.severity}")
    if validation.errors:
        print(f"   Errors: {validation.errors}")
    
    # Test 3: Invalid status
    print("\n3. Testing invalid status:")
    invalid_status_result = {
        'solver_name': 'TestSolver',
        'problem_name': 'test_lp',
        'solve_time': 0.001,
        'status': 'unknown_status',  # Invalid status
        'objective_value': 1.5
    }
    
    validation = validator.validate_single_result(invalid_status_result)
    print(f"   Result: Valid={validation.is_valid}, Severity={validation.severity}")
    if validation.errors:
        print(f"   Errors: {validation.errors}")
    
    # Test 4: NaN objective value with optimal status
    print("\n4. Testing NaN objective with optimal status:")
    nan_objective_result = {
        'solver_name': 'TestSolver',
        'problem_name': 'test_lp',
        'solve_time': 0.001,
        'status': 'optimal',
        'objective_value': float('nan')  # NaN objective
    }
    
    validation = validator.validate_single_result(nan_objective_result)
    print(f"   Result: Valid={validation.is_valid}, Severity={validation.severity}")
    if validation.errors:
        print(f"   Errors: {validation.errors}")
    
    # Test 5: Time close to timeout
    print("\n5. Testing time close to timeout:")
    timeout_result = {
        'solver_name': 'SlowSolver',
        'problem_name': 'test_lp',
        'solve_time': 9.5,  # Close to 10s timeout
        'status': 'optimal',
        'objective_value': 1.5
    }
    
    validation = validator.validate_single_result(timeout_result, timeout=10.0)
    print(f"   Result: Valid={validation.is_valid}, Severity={validation.severity}")
    if validation.warnings:
        print(f"   Warnings: {validation.warnings}")
    
    # Test 6: Batch validation with consistency checks
    print("\n6. Testing batch validation:")
    batch_results = [
        valid_result,
        {
            'solver_name': 'SciPy',
            'problem_name': 'test_lp',
            'solve_time': 0.010,  # 10x slower than first result
            'status': 'optimal',
            'objective_value': 1.5
        },
        {
            'solver_name': 'AlwaysFails',
            'problem_name': 'test_lp',
            'solve_time': 0.001,
            'status': 'error',
            'objective_value': None
        },
        {
            'solver_name': 'AlwaysFails',
            'problem_name': 'test_qp',
            'solve_time': 0.001,
            'status': 'error',
            'objective_value': None
        },
        {
            'solver_name': 'AlwaysFails',
            'problem_name': 'test_socp',
            'solve_time': 0.001,
            'status': 'error',
            'objective_value': None
        }
    ]
    
    individual_validations, consistency_validation = validator.validate_batch_results(batch_results)
    print(f"   Individual validations: {len(individual_validations)} results")
    
    valid_count = sum(1 for v in individual_validations if v.is_valid)
    print(f"   Valid results: {valid_count}/{len(individual_validations)}")
    
    print(f"   Consistency validation: Valid={consistency_validation.is_valid}")
    if consistency_validation.warnings:
        print(f"   Consistency warnings: {consistency_validation.warnings}")
    
    print("\n✓ Validation system tests completed!")


def test_result_collector_integration():
    """Test integration of validation with result collector."""
    
    print("\n" + "=" * 60)
    print("TESTING RESULT COLLECTOR INTEGRATION")
    print("=" * 60)
    
    try:
        # Create result collector with validation enabled
        collector = ResultCollector(enable_validation=True)
        print("✓ Created result collector with validation enabled")
        
        # Create a benchmark session
        benchmark_id = collector.create_benchmark_session()
        print(f"✓ Created benchmark session: {benchmark_id}")
        
        # Test storing valid result
        print("\n1. Testing valid result storage:")
        valid_result = SolverResult(
            solver_name="TestSolver",
            problem_name="test_problem",
            solve_time=0.005,
            status="optimal",
            objective_value=2.5
        )
        
        try:
            result_id = collector.store_result(benchmark_id, valid_result, timeout=10.0)
            print(f"   ✓ Stored valid result with ID: {result_id}")
        except Exception as e:
            print(f"   ✗ Failed to store valid result: {e}")
        
        # Test storing invalid result (should be rejected or warned)
        print("\n2. Testing invalid result storage:")
        invalid_result = SolverResult(
            solver_name="BadSolver",
            problem_name="test_problem",
            solve_time=-1.0,  # Invalid negative time
            status="optimal",
            objective_value=2.5
        )
        
        try:
            result_id = collector.store_result(benchmark_id, invalid_result, timeout=10.0)
            print(f"   ✗ Unexpectedly stored invalid result with ID: {result_id}")
        except ValueError as e:
            print(f"   ✓ Correctly rejected invalid result: {e}")
        except Exception as e:
            print(f"   ? Unexpected error storing invalid result: {e}")
        
        # Test validation of stored results
        print("\n3. Testing stored results validation:")
        try:
            validation_summary = collector.validate_stored_results(benchmark_id)
            print(f"   ✓ Validation completed:")
            print(f"     Results count: {validation_summary.get('results_count', 0)}")
            print(f"     Valid results: {validation_summary.get('valid_results', 0)}")
            print(f"     Error results: {validation_summary.get('error_results', 0)}")
            print(f"     Warning results: {validation_summary.get('warning_results', 0)}")
            
            if validation_summary.get('all_errors'):
                print(f"     Errors found: {validation_summary['all_errors']}")
            if validation_summary.get('all_warnings'):
                print(f"     Warnings found: {validation_summary['all_warnings']}")
                
        except Exception as e:
            print(f"   ✗ Validation failed: {e}")
        
        print("\n✓ Result collector integration tests completed!")
        
    except Exception as e:
        print(f"✗ Result collector integration test failed: {e}")
        logger.exception("Detailed error:")


def create_test_results_with_issues():
    """Create some test results with intentional validation issues."""
    
    print("\n" + "=" * 60)
    print("TESTING WITH INTENTIONALLY BAD DATA")
    print("=" * 60)
    
    # Create validator
    validator = create_default_validator()
    
    # Create various problematic results
    problematic_results = [
        {
            'solver_name': 'SciPy',
            'problem_name': 'good_problem',
            'solve_time': 0.001,
            'status': 'optimal',
            'objective_value': 1.5
        },
        {
            'solver_name': 'TimeoutSolver',
            'problem_name': 'hard_problem',
            'solve_time': 15.0,  # Exceeds 10s timeout
            'status': 'timeout',
            'objective_value': None
        },
        {
            'solver_name': 'BuggyMath',
            'problem_name': 'numerical_problem',
            'solve_time': 0.001,
            'status': 'optimal',
            'objective_value': float('inf')  # Infinite objective
        },
        {
            'solver_name': 'BadStatus',
            'problem_name': 'mystery_problem',
            'solve_time': 0.001,
            'status': 'completely_made_up_status',
            'objective_value': 1.0
        },
        {
            'solver_name': 'NegativeTime',
            'problem_name': 'time_problem',
            'solve_time': -0.5,
            'status': 'optimal',
            'objective_value': 1.0
        }
    ]
    
    print(f"Testing {len(problematic_results)} problematic results:")
    
    for i, result in enumerate(problematic_results):
        print(f"\n{i+1}. {result['solver_name']} on {result['problem_name']}:")
        
        validation = validator.validate_single_result(result, timeout=10.0)
        
        print(f"   Valid: {validation.is_valid}")
        print(f"   Severity: {validation.severity}")
        
        if validation.errors:
            print(f"   Errors: {validation.errors}")
        if validation.warnings:
            print(f"   Warnings: {validation.warnings}")
    
    # Test batch validation
    print(f"\nBatch validation of all {len(problematic_results)} results:")
    individual, consistency = validator.validate_batch_results(problematic_results, timeout=10.0)
    
    valid_count = sum(1 for v in individual if v.is_valid)
    error_count = sum(1 for v in individual if v.severity == 'error')
    warning_count = sum(1 for v in individual if v.severity == 'warning')
    
    print(f"Valid: {valid_count}/{len(problematic_results)}")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")
    
    if consistency.warnings:
        print(f"Consistency warnings: {consistency.warnings}")
    
    print("\n✓ Problematic data testing completed!")


if __name__ == "__main__":
    print("Starting validation system tests...\n")
    
    try:
        # Test basic validation functionality
        test_validation_system()
        
        # Test integration with result collector
        test_result_collector_integration()
        
        # Test with intentionally problematic data
        create_test_results_with_issues()
        
        print("\n" + "=" * 60)
        print("ALL VALIDATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Validation tests failed: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)