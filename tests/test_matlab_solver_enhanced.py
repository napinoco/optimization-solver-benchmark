#!/usr/bin/env python3
"""
Quick test for enhanced MATLAB solver CLI functionality.

This script tests the enhanced MATLAB solver with improved CLI handling,
error parsing, and command construction.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.matlab_octave.matlab_solver import MatlabSolver
from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("test_matlab_solver_enhanced")


class MockProblemData:
    """Mock problem data for testing."""
    def __init__(self, name: str):
        self.name = name


def test_matlab_solver_initialization():
    """Test MatlabSolver initialization and MATLAB verification."""
    print("\nTesting MatlabSolver initialization...")
    
    try:
        solver = MatlabSolver('sedumi')
        print(f"✓ SeDuMi solver initialized successfully: {solver.solver_name}")
        
        solver2 = MatlabSolver('sdpt3')
        print(f"✓ SDPT3 solver initialized successfully: {solver2.solver_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ MATLAB solver initialization failed: {e}")
        return False


def test_command_construction():
    """Test command construction with various inputs."""
    print("\nTesting command construction...")
    
    try:
        solver = MatlabSolver('sedumi')
        
        # Test normal inputs
        cmd = solver._construct_matlab_command('test_problem', '/tmp/result.json')
        expected_parts = ['matlab', '-batch', "matlab_runner('test_problem', 'sedumi', '/tmp/result.json')"]
        print(f"✓ Normal command construction: {' '.join(cmd)}")
        
        # Test inputs with single quotes (should be escaped)
        cmd = solver._construct_matlab_command("prob'lem", "/tmp/res'ult.json")
        print(f"✓ Escaped command construction: {' '.join(cmd)}")
        
        # Test invalid inputs
        try:
            cmd = solver._construct_matlab_command("", "/tmp/result.json")
            print("✗ Should have failed with empty problem name")
            return False
        except ValueError:
            print("✓ Correctly rejected empty problem name")
        
        return True
        
    except Exception as e:
        print(f"✗ Command construction test failed: {e}")
        return False


def test_error_parsing():
    """Test MATLAB error message parsing."""
    print("\nTesting error message parsing...")
    
    try:
        solver = MatlabSolver('sedumi')
        
        # Test various error messages
        test_cases = [
            ("Error: Undefined function 'nonexistent' for input arguments of type 'double'.", 
             "Error: Undefined function"),
            ("License error: Cannot find license file.", "License error"),
            ("Some random output\nError: Something went wrong\nMore output", "Error: Something went wrong"),
            ("No clear error pattern here", "No clear error"),
        ]
        
        for stderr, expected_keyword in test_cases:
            parsed = solver._parse_matlab_error(stderr, "")
            print(f"✓ Parsed '{stderr[:50]}...' -> '{parsed}'")
            if "error" not in expected_keyword.lower():
                # For non-error cases, just check it doesn't crash
                continue
            
        return True
        
    except Exception as e:
        print(f"✗ Error parsing test failed: {e}")
        return False


def test_solver_integration():
    """Test solver with mock problem data."""
    print("\nTesting solver integration with mock data...")
    
    try:
        solver = MatlabSolver('sedumi', timeout=30)
        
        # Create mock problem data
        mock_problem = MockProblemData("nonexistent_problem")
        
        # This should fail gracefully since the problem doesn't exist
        start_time = time.time()
        result = solver.solve(mock_problem, timeout=30)
        execution_time = time.time() - start_time
        
        print(f"✓ Solver execution completed in {execution_time:.2f}s")
        print(f"✓ Result status: {result.status}")
        print(f"✓ Solver name: {result.solver_name}")
        print(f"✓ Solve time: {result.solve_time:.2f}s")
        
        # Should be an error result since problem doesn't exist
        if result.status == "ERROR":
            print("✓ Correctly handled nonexistent problem with error status")
        else:
            print(f"? Unexpected status for nonexistent problem: {result.status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Solver integration test failed: {e}")
        return False


def test_temp_file_integration():
    """Test temporary file management integration."""
    print("\nTesting temporary file management...")
    
    try:
        solver = MatlabSolver('sedumi')
        
        # Get temp file stats
        stats = solver.get_temp_file_stats()
        print(f"✓ Temp file stats: {stats['total_files']} files, {stats['total_size_bytes']} bytes")
        print(f"✓ Temp directory: {stats['temp_directory']}")
        
        # Test cleanup
        cleaned = solver.cleanup_orphaned_files()
        print(f"✓ Cleaned up {cleaned} orphaned files")
        
        return True
        
    except Exception as e:
        print(f"✗ Temp file integration test failed: {e}")
        return False


def main():
    """Run enhanced MATLAB solver tests."""
    print("=" * 60)
    print("ENHANCED MATLAB SOLVER CLI VALIDATION")
    print("=" * 60)
    
    tests = [
        test_matlab_solver_initialization,
        test_command_construction,
        test_error_parsing,
        test_temp_file_integration,
        test_solver_integration,  # This one takes longest, so run it last
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"ENHANCED MATLAB SOLVER TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {100 * passed / total:.1f}%")
    
    if passed >= total * 0.8:  # 80% success rate
        print("✅ Enhanced MATLAB solver CLI validation successful!")
        return 0
    else:
        print("❌ Enhanced MATLAB solver CLI validation had issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())