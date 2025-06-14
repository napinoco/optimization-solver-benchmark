#!/usr/bin/env python3
"""
Test Simplified Octave Integration
==================================

Test the simplified octave_runner_simple.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.solvers.octave.octave_runner_simple import OctaveSolverSimple
from scripts.solvers.octave.test_octave import create_simple_lp, create_simple_qp

def test_simple_solver():
    """Test the simplified Octave solver."""
    
    print("🧪 Testing Simplified Octave Solver Integration")
    print("=" * 50)
    
    try:
        # Create solver
        solver = OctaveSolverSimple()
        print(f"✅ Solver created: {solver.name}")
        print(f"📍 Octave path: {solver.octave_path}")
        print(f"🔧 Version: {solver.get_version()}")
        print(f"🟢 Available: {solver.is_available()}")
        
        # Test LP
        print("\n🧮 Testing Linear Programming...")
        lp_problem = create_simple_lp()
        lp_result = solver.solve(lp_problem)
        
        print(f"  Problem: {lp_problem.name}")
        print(f"  Status: {lp_result.status}")
        print(f"  Objective: {lp_result.objective_value}")
        print(f"  Solve time: {lp_result.solve_time:.4f}s")
        
        if lp_result.status == 'optimal':
            print("  ✅ LP test PASSED!")
            lp_success = True
        else:
            print(f"  ❌ LP test FAILED: {lp_result.error_message}")
            lp_success = False
        
        # Test QP
        print("\n🔄 Testing Quadratic Programming...")
        qp_problem = create_simple_qp()
        qp_result = solver.solve(qp_problem)
        
        print(f"  Problem: {qp_problem.name}")
        print(f"  Status: {qp_result.status}")
        print(f"  Objective: {qp_result.objective_value}")
        print(f"  Solve time: {qp_result.solve_time:.4f}s")
        
        if qp_result.status == 'optimal':
            print("  ✅ QP test PASSED!")
            qp_success = True
        else:
            print(f"  ❌ QP test FAILED: {qp_result.error_message}")
            qp_success = False
        
        # Summary
        print(f"\n📊 Test Results")
        print("=" * 20)
        if lp_success and qp_success:
            print("🎉 All tests PASSED! Simplified integration is working.")
            return True
        else:
            print("❌ Some tests failed.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_solver()
    exit(0 if success else 1)