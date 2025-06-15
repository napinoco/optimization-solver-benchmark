#!/usr/bin/env python3
"""
Debug Python Integration
=========================

Test the Python-based optimization solving integration.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_python_integration():
    """Test the Python optimization solving integration."""
    
    print("🔍 Testing Python Integration...")
    
    try:
        from scripts.benchmark.problem_loader import load_problem
        
        # Test loading a problem from the registry
        print("📝 Loading problem from registry...")
        problem = load_problem("simple_lp_001", "light_set")
        print(f"✅ Problem loaded: {problem.name}")
        print(f"   Class: {problem.problem_class}")
        print(f"   Variables: {len(problem.c) if problem.c is not None else 0}")
        
        # Test basic problem structure
        if problem.c is not None:
            print(f"   Objective coefficients: {problem.c}")
        if problem.A_ub is not None:
            print(f"   Constraints shape: {problem.A_ub.shape}")
            print(f"   RHS: {problem.b_ub}")
        
        print("🎉 Python integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_python_integration()
    exit(0 if success else 1)