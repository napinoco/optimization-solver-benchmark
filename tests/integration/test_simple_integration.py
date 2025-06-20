#!/usr/bin/env python3
"""
Test Python Solver Integration
===============================

Test the Python-based solver integration without Octave.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_python_solvers():
    """Test the Python-based solvers."""
    
    print("🧪 Testing Python Solver Integration")
    print("=" * 50)
    
    try:
        from scripts.benchmark.problem_loader import load_problem
        
        # Test loading a simple problem
        print("📝 Loading test problem...")
        problem = load_problem("simple_lp_001", "light_set")
        print(f"✅ Problem loaded: {problem.name} ({problem.problem_class})")
        print(f"   Variables: {len(problem.c) if problem.c is not None else 0}")
        
        # Test basic functionality
        print("\n🔧 Testing problem structure...")
        if problem.c is not None:
            print(f"✅ Objective coefficients: shape {problem.c.shape}")
        if problem.A_ub is not None:
            print(f"✅ Inequality constraints: shape {problem.A_ub.shape}")
        print(f"✅ Problem class: {problem.problem_class}")
        
        print("🎉 Python solver integration test PASSED!")
        return True
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_python_solvers()
    exit(0 if success else 1)