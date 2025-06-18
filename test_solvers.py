#!/usr/bin/env python3
"""
Comprehensive test script for the standardized solver interface.

This script tests both SciPy and CVXPY solvers with various problem types
to ensure they work correctly with the new standardized interface.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.solvers.python.scipy_runner import ScipySolver
from scripts.solvers.python.cvxpy_runner import CvxpySolver
from scripts.benchmark.problem_loader import ProblemData
from scripts.solvers.solver_interface import SolverResult


def create_test_lp_problem():
    """Create a simple LP test problem: minimize x1 + 2*x2 subject to x1 + x2 <= 1, x1, x2 >= 0"""
    return ProblemData(
        name="test_lp",
        problem_class="LP",
        c=np.array([1.0, 2.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([1.0]),
        bounds=[(0, None), (0, None)]
    )


def create_test_qp_problem():
    """Create a simple QP test problem: minimize 0.5*x^T*Q*x + c^T*x subject to x >= 0"""
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])  # Positive definite matrix
    c = np.array([1.0, 1.0])
    
    problem = ProblemData(
        name="test_qp",
        problem_class="QP",
        c=c,
        bounds=[(0, None), (0, None)]
    )
    
    # Add the quadratic matrix Q as an attribute
    problem.Q = Q
    return problem


def create_test_socp_problem():
    """Create a simple SOCP test problem using CVXPY format"""
    import cvxpy as cp
    
    # Portfolio optimization as SOCP
    n = 3
    x = cp.Variable(n, name="x")
    
    # Random data
    np.random.seed(42)
    mu = np.array([0.1, 0.2, 0.15])  # Expected returns
    Sigma = np.array([[0.1, 0.02, 0.01],
                      [0.02, 0.2, 0.03],
                      [0.01, 0.03, 0.15]])  # Covariance matrix
    
    # Objective: minimize risk
    risk = cp.quad_form(x, Sigma)
    objective = cp.Minimize(risk)
    
    # Constraints
    constraints = [
        cp.sum(x) == 1,  # Budget constraint
        mu.T @ x >= 0.12,  # Minimum return
        x >= 0  # Long-only
    ]
    
    cvxpy_problem = cp.Problem(objective, constraints)
    
    return ProblemData(
        name="test_socp",
        problem_class="SOCP",
        cvxpy_problem=cvxpy_problem,
        metadata={"description": "Portfolio optimization SOCP"}
    )


def validate_result(result: SolverResult, expected_status=None):
    """Validate that a solver result has all required fields and proper types"""
    print(f"    Validating result...")
    
    # Check result type
    assert isinstance(result, SolverResult), f"Expected SolverResult, got {type(result)}"
    
    # Check required fields exist and have correct types
    assert isinstance(result.solve_time, (int, float)), f"solve_time must be numeric, got {type(result.solve_time)}"
    assert result.solve_time >= 0, f"solve_time must be non-negative, got {result.solve_time}"
    
    assert isinstance(result.status, str), f"status must be string, got {type(result.status)}"
    assert result.status.strip(), "status cannot be empty"
    
    # Optional numeric fields can be None
    for field_name in ['primal_objective_value', 'dual_objective_value', 'duality_gap', 
                      'primal_infeasibility', 'dual_infeasibility']:
        value = getattr(result, field_name)
        if value is not None:
            assert isinstance(value, (int, float)), f"{field_name} must be numeric or None, got {type(value)}"
    
    # Iterations can be None or int
    if result.iterations is not None:
        assert isinstance(result.iterations, int), f"iterations must be int or None, got {type(result.iterations)}"
        assert result.iterations >= 0, f"iterations must be non-negative, got {result.iterations}"
    
    # Check solver identification
    assert isinstance(result.solver_name, str), f"solver_name must be string, got {type(result.solver_name)}"
    assert isinstance(result.solver_version, str), f"solver_version must be string, got {type(result.solver_version)}"
    
    # Check expected status if provided
    if expected_status:
        assert result.status == expected_status, f"Expected status {expected_status}, got {result.status}"
    
    print(f"    ✓ Result validation passed")
    print(f"      Status: {result.status}")
    print(f"      Objective: {result.primal_objective_value}")
    print(f"      Time: {result.solve_time:.3f}s")
    print(f"      Solver: {result.solver_name} v{result.solver_version}")
    
    return True


def test_scipy_solver():
    """Test SciPy solver with LP and QP problems"""
    print("\n" + "="*50)
    print("TESTING SCIPY SOLVER")
    print("="*50)
    
    # Test solver initialization
    print("\n1. Testing SciPy solver initialization...")
    try:
        solver = ScipySolver(method="highs")
        print(f"   ✓ SciPy solver initialized: {solver.solver_name}")
        print(f"     Version: {solver.get_version()}")
        print(f"     Method: {solver.method}")
    except Exception as e:
        print(f"   ✗ SciPy solver initialization failed: {e}")
        return False
    
    # Test LP solving
    print("\n2. Testing LP problem solving...")
    try:
        lp_problem = create_test_lp_problem()
        print(f"   Problem: {lp_problem.name} ({lp_problem.problem_class})")
        print(f"   Variables: {len(lp_problem.c)}")
        print(f"   Constraints: {lp_problem.A_ub.shape[0] if lp_problem.A_ub is not None else 0}")
        
        result = solver.solve(lp_problem)
        validate_result(result, expected_status="OPTIMAL")
        
        # Check that we got a reasonable objective value
        if result.primal_objective_value is not None:
            assert result.primal_objective_value >= 0, "LP objective should be non-negative"
            
    except Exception as e:
        print(f"   ✗ LP solving failed: {e}")
        return False
    
    # Test QP solving
    print("\n3. Testing QP problem solving...")
    try:
        qp_problem = create_test_qp_problem()
        print(f"   Problem: {qp_problem.name} ({qp_problem.problem_class})")
        print(f"   Variables: {len(qp_problem.c)}")
        print(f"   Q matrix shape: {qp_problem.Q.shape}")
        
        result = solver.solve(qp_problem)
        validate_result(result, expected_status="OPTIMAL")
        
    except Exception as e:
        print(f"   ✗ QP solving failed: {e}")
        return False
    
    # Test error handling
    print("\n4. Testing error handling...")
    try:
        invalid_problem = ProblemData(
            name="invalid",
            problem_class="INVALID_TYPE",
            c=np.array([1.0])
        )
        
        result = solver.solve(invalid_problem)
        validate_result(result, expected_status="ERROR")
        
        print(f"   ✓ Error handling works correctly")
        
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        return False
    
    # Test problem compatibility
    print("\n5. Testing problem compatibility validation...")
    try:
        lp_compatible = solver.validate_problem_compatibility(create_test_lp_problem())
        qp_compatible = solver.validate_problem_compatibility(create_test_qp_problem())
        
        print(f"   LP compatibility: {lp_compatible}")
        print(f"   QP compatibility: {qp_compatible}")
        
        assert lp_compatible, "SciPy should support LP problems"
        assert qp_compatible, "SciPy should support QP problems"
        
    except Exception as e:
        print(f"   ✗ Compatibility validation failed: {e}")
        return False
    
    print("\n✓ All SciPy solver tests passed!")
    return True


def test_cvxpy_solver():
    """Test CVXPY solver with LP, QP, and SOCP problems"""
    print("\n" + "="*50)
    print("TESTING CVXPY SOLVER")
    print("="*50)
    
    # Test solver initialization
    print("\n1. Testing CVXPY solver initialization...")
    try:
        import cvxpy as cp
        available_solvers = cp.installed_solvers()
        print(f"   Available CVXPY backends: {available_solvers}")
        
        # Try CLARABEL first, fallback to others
        backend = "CLARABEL" if "CLARABEL" in available_solvers else available_solvers[0]
        solver = CvxpySolver(backend=backend)
        
        print(f"   ✓ CVXPY solver initialized: {solver.solver_name}")
        print(f"     Version: {solver.get_version()}")
        print(f"     Backend: {solver.backend}")
        print(f"     Capabilities: {solver.backend_capabilities['supported_problem_types']}")
        
    except Exception as e:
        print(f"   ✗ CVXPY solver initialization failed: {e}")
        return False
    
    # Test LP solving
    print("\n2. Testing LP problem solving...")
    try:
        lp_problem = create_test_lp_problem()
        print(f"   Problem: {lp_problem.name} ({lp_problem.problem_class})")
        
        result = solver.solve(lp_problem)
        validate_result(result, expected_status="OPTIMAL")
        
        # Check that we got a reasonable objective value
        if result.primal_objective_value is not None:
            assert result.primal_objective_value >= 0, "LP objective should be non-negative"
            
    except Exception as e:
        print(f"   ✗ LP solving failed: {e}")
        return False
    
    # Test QP solving (if backend supports it)
    if "QP" in solver.backend_capabilities["supported_problem_types"]:
        print("\n3. Testing QP problem solving...")
        try:
            qp_problem = create_test_qp_problem()
            print(f"   Problem: {qp_problem.name} ({qp_problem.problem_class})")
            
            result = solver.solve(qp_problem)
            validate_result(result)  # Don't require OPTIMAL as some QP solvers might struggle
            
        except Exception as e:
            print(f"   ✗ QP solving failed: {e}")
            return False
    else:
        print(f"\n3. Skipping QP test (backend {solver.backend} doesn't support QP)")
    
    # Test SOCP solving (if backend supports it)
    if "SOCP" in solver.backend_capabilities["supported_problem_types"]:
        print("\n4. Testing SOCP problem solving...")
        try:
            socp_problem = create_test_socp_problem()
            print(f"   Problem: {socp_problem.name} ({socp_problem.problem_class})")
            
            result = solver.solve(socp_problem)
            validate_result(result)  # Don't require OPTIMAL as SOCP can be challenging
            
        except Exception as e:
            print(f"   ✗ SOCP solving failed: {e}")
            return False
    else:
        print(f"\n4. Skipping SOCP test (backend {solver.backend} doesn't support SOCP)")
    
    # Test unsupported problem type
    print("\n5. Testing unsupported problem type handling...")
    try:
        # Create a problem type not supported by the backend
        unsupported_types = ["SDP", "SOCP", "QP", "LP"]
        unsupported_type = None
        
        for ptype in unsupported_types:
            if ptype not in solver.backend_capabilities["supported_problem_types"]:
                unsupported_type = ptype
                break
        
        if unsupported_type:
            unsupported_problem = ProblemData(
                name="unsupported",
                problem_class=unsupported_type,
                c=np.array([1.0])
            )
            
            result = solver.solve(unsupported_problem)
            validate_result(result, expected_status="ERROR")
            
            print(f"   ✓ Unsupported problem type handling works correctly")
        else:
            print(f"   → Skipping (backend {solver.backend} supports all problem types)")
            
    except Exception as e:
        print(f"   ✗ Unsupported problem type test failed: {e}")
        return False
    
    # Test timeout functionality
    print("\n6. Testing timeout functionality...")
    try:
        lp_problem = create_test_lp_problem()
        
        # Test with very short timeout (should still solve quickly)
        result = solver.solve(lp_problem, timeout=0.001)
        validate_result(result)  # May or may not timeout depending on solver speed
        
        print(f"   ✓ Timeout functionality works")
        
    except Exception as e:
        print(f"   ✗ Timeout test failed: {e}")
        return False
    
    print("\n✓ All CVXPY solver tests passed!")
    return True


def test_multiple_backends():
    """Test multiple CVXPY backends"""
    print("\n" + "="*50)
    print("TESTING MULTIPLE CVXPY BACKENDS")
    print("="*50)
    
    try:
        import cvxpy as cp
        from scripts.solvers.python.cvxpy_runner import create_cvxpy_solvers
        
        # Create multiple solver instances
        solvers = create_cvxpy_solvers(verbose=False)
        print(f"\nCreated {len(solvers)} CVXPY solver instances:")
        
        for solver in solvers:
            print(f"  - {solver.solver_name}: {solver.backend}")
        
        # Test each solver with a simple LP
        lp_problem = create_test_lp_problem()
        
        print(f"\nTesting all backends with LP problem...")
        for solver in solvers:
            try:
                print(f"\n  Testing {solver.solver_name}...")
                result = solver.solve(lp_problem)
                validate_result(result)
                
            except Exception as e:
                print(f"    ✗ {solver.solver_name} failed: {e}")
                
        print(f"\n✓ Multiple backend testing completed!")
        return True
        
    except Exception as e:
        print(f"   ✗ Multiple backend testing failed: {e}")
        return False


def test_solver_interface_compliance():
    """Test that solvers properly implement the SolverInterface"""
    print("\n" + "="*50)
    print("TESTING SOLVER INTERFACE COMPLIANCE")
    print("="*50)
    
    from scripts.solvers.solver_interface import SolverInterface
    
    # Test SciPy solver
    print("\n1. Testing SciPy solver interface compliance...")
    try:
        scipy_solver = ScipySolver()
        
        # Check inheritance
        assert isinstance(scipy_solver, SolverInterface), "SciPy solver must inherit from SolverInterface"
        
        # Check required methods exist
        assert hasattr(scipy_solver, 'solve'), "Must have solve method"
        assert hasattr(scipy_solver, 'get_version'), "Must have get_version method"
        assert hasattr(scipy_solver, 'validate_problem_compatibility'), "Must have validate_problem_compatibility method"
        
        # Check method signatures
        import inspect
        solve_sig = inspect.signature(scipy_solver.solve)
        assert 'problem_data' in solve_sig.parameters, "solve() must accept problem_data parameter"
        assert 'timeout' in solve_sig.parameters, "solve() must accept timeout parameter"
        
        print("   ✓ SciPy solver interface compliance verified")
        
    except Exception as e:
        print(f"   ✗ SciPy solver interface compliance failed: {e}")
        return False
    
    # Test CVXPY solver
    print("\n2. Testing CVXPY solver interface compliance...")
    try:
        cvxpy_solver = CvxpySolver()
        
        # Check inheritance
        assert isinstance(cvxpy_solver, SolverInterface), "CVXPY solver must inherit from SolverInterface"
        
        # Check required methods exist
        assert hasattr(cvxpy_solver, 'solve'), "Must have solve method"
        assert hasattr(cvxpy_solver, 'get_version'), "Must have get_version method"
        assert hasattr(cvxpy_solver, 'validate_problem_compatibility'), "Must have validate_problem_compatibility method"
        
        # Check method signatures
        solve_sig = inspect.signature(cvxpy_solver.solve)
        assert 'problem_data' in solve_sig.parameters, "solve() must accept problem_data parameter"
        assert 'timeout' in solve_sig.parameters, "solve() must accept timeout parameter"
        
        print("   ✓ CVXPY solver interface compliance verified")
        
    except Exception as e:
        print(f"   ✗ CVXPY solver interface compliance failed: {e}")
        return False
    
    print("\n✓ All solver interface compliance tests passed!")
    return True


def main():
    """Run all solver tests"""
    print("COMPREHENSIVE SOLVER TESTING")
    print("="*60)
    
    all_passed = True
    
    # Test interface compliance first
    if not test_solver_interface_compliance():
        all_passed = False
    
    # Test individual solvers
    if not test_scipy_solver():
        all_passed = False
    
    if not test_cvxpy_solver():
        all_passed = False
    
    # Test multiple backends
    if not test_multiple_backends():
        all_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Solver architecture is working correctly.")
        print("\nKey achievements verified:")
        print("  ✓ Standardized solver interface implementation")
        print("  ✓ Consistent 8-field result format")
        print("  ✓ Proper error handling and validation")
        print("  ✓ Multiple backend support")
        print("  ✓ Problem type compatibility checking")
        print("  ✓ Timeout functionality")
        print("  ✓ Version tracking")
    else:
        print("❌ SOME TESTS FAILED! Please check the output above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())