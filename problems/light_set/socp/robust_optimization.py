"""
Robust Optimization SOCP Problem
=================================

Robust linear programming with ellipsoidal uncertainty sets formulated as SOCP.

Problem formulation:
    minimize    c^T * x
    subject to  ||u_i||_2 <= t_i                [SOCP constraint for each uncertain constraint]
                (A_i * x - b_i) + P_i^T * u_i = 0   [Uncertainty coupling]
                x >= 0                           [Non-negativity]

where:
- x: decision variables
- c: objective coefficients
- A_i, b_i: nominal constraint data
- P_i: uncertainty matrix (defines ellipsoidal uncertainty set)
- u_i: uncertainty variables
- t_i: uncertainty bounds

This formulation handles uncertainty in constraint coefficients using ellipsoidal sets.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData


def generate_robust_optimization_socp(n_vars: int = 4, n_constraints: int = 3, seed: int = 42) -> Tuple[Dict[str, Any], ProblemData]:
    """
    Generate a robust optimization SOCP problem.
    
    Args:
        n_vars: Number of decision variables
        n_constraints: Number of uncertain constraints
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problem_dict, ProblemData)
    """
    np.random.seed(seed)
    
    # Objective coefficients (we want to minimize cost)
    c = np.random.rand(n_vars)
    
    # Generate nominal constraint data A*x <= b
    A_nominal = np.random.rand(n_constraints, n_vars)
    b_nominal = np.random.rand(n_constraints) + 1.0  # Ensure feasible nominal problem
    
    # Generate uncertainty matrices (ellipsoidal uncertainty sets)
    uncertainty_matrices = []
    uncertainty_bounds = []
    
    for i in range(n_constraints):
        # Create uncertainty matrix P_i (determines shape of uncertainty ellipsoid)
        P_i = 0.1 * np.random.randn(n_vars, 2)  # 2D uncertainty for each constraint
        uncertainty_matrices.append(P_i)
        
        # Uncertainty bound (radius of ellipsoid)
        t_i = 0.5 + 0.3 * np.random.rand()
        uncertainty_bounds.append(t_i)
    
    # Decision variables
    x = cp.Variable(n_vars, name="decision_vars")
    
    # Uncertainty variables and bounds
    u_vars = []
    t_vars = []
    for i in range(n_constraints):
        u_i = cp.Variable(2, name=f"uncertainty_{i}")  # 2D uncertainty for each constraint
        t_i = cp.Variable(name=f"uncertainty_bound_{i}")
        u_vars.append(u_i)
        t_vars.append(t_i)
    
    # Objective: minimize cost under worst-case uncertainty
    objective = cp.Minimize(c.T @ x)
    
    # Constraints
    constraints = []
    
    # SOCP constraints for uncertainty sets: ||u_i||_2 <= t_i
    for i in range(n_constraints):
        constraints.append(cp.SOC(t_vars[i], u_vars[i]))
    
    # Robust constraints: A_i*x - b_i + P_i^T*u_i <= 0 becomes worst-case constraint
    for i in range(n_constraints):
        P_i = uncertainty_matrices[i]
        # Robust constraint: A_i*x + ||P_i^T||_2 * t_i <= b_i
        robust_constraint = A_nominal[i, :] @ x + cp.norm(P_i, 2) * t_vars[i] <= b_nominal[i]
        constraints.append(robust_constraint)
    
    # Bound uncertainty variables by their ellipsoid radius
    for i in range(n_constraints):
        constraints.append(t_vars[i] <= uncertainty_bounds[i])
        constraints.append(t_vars[i] >= 0)
    
    # Non-negativity constraints
    constraints.append(x >= 0)
    
    # Create CVXPY problem
    problem = cp.Problem(objective, constraints)
    
    # Create problem data structure
    problem_dict = {
        "name": "robust_optimization",
        "description": "Robust optimization with ellipsoidal uncertainty",
        "problem_class": "SOCP",
        "n_variables": n_vars + sum(len(u_i.shape) if hasattr(u_i.shape, '__len__') else 1 for u_i in u_vars) + len(t_vars),
        "n_constraints": len(constraints),
        "n_soc_constraints": n_constraints,
        "objective_sense": "minimize",
        "nominal_constraint_matrix": A_nominal.tolist(),
        "nominal_rhs": b_nominal.tolist(),
        "uncertainty_matrices": [P.tolist() for P in uncertainty_matrices],
        "uncertainty_bounds": uncertainty_bounds,
        "objective_coefficients": c.tolist(),
        "cvxpy_problem": problem,
        "variables": {"x": x, "u_vars": u_vars, "t_vars": t_vars}
    }
    
    # Create ProblemData instance
    problem_data = ProblemData(
        name="robust_optimization",
        problem_class="SOCP",
        cvxpy_problem=problem,
        variables={"x": x, "u_vars": u_vars, "t_vars": t_vars},
        objective=objective,
        constraints=constraints,
        metadata={
            "description": "Robust optimization with ellipsoidal uncertainty",
            "n_vars": n_vars,
            "n_constraints": n_constraints,
            "n_soc_constraints": n_constraints,
            "uncertainty_level": "ellipsoidal",
            "nominal_data": {
                "A": A_nominal,
                "b": b_nominal,
                "c": c
            },
            "uncertainty_data": {
                "matrices": uncertainty_matrices,
                "bounds": uncertainty_bounds
            }
        }
    )
    
    return problem_dict, problem_data


def solve_robust_optimization(solver_name: str = "CLARABEL") -> Dict[str, Any]:
    """Solve the robust optimization problem and return results."""
    
    problem_dict, problem_data = generate_robust_optimization_socp()
    problem = problem_dict["cvxpy_problem"]
    
    # Solve the problem
    try:
        problem.solve(solver=getattr(cp, solver_name), verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            x_opt = problem_dict["variables"]["x"].value
            
            # Calculate robustness metrics
            c = np.array(problem_dict["objective_coefficients"])
            optimal_cost = c.T @ x_opt
            
            results = {
                "status": problem.status,
                "optimal_value": problem.value,
                "optimal_solution": x_opt.tolist(),
                "robust_cost": float(optimal_cost),
                "uncertainty_bounds": problem_dict["uncertainty_bounds"],
                "solve_time": problem.solver_stats.solve_time if problem.solver_stats else None
            }
        else:
            results = {
                "status": problem.status,
                "optimal_value": None,
                "error": "Problem not solved optimally"
            }
            
    except Exception as e:
        results = {
            "status": "error",
            "error": str(e)
        }
    
    return results


if __name__ == "__main__":
    # Test the robust optimization problem
    print("Testing Robust Optimization SOCP Problem...")
    
    # Generate problem
    problem_dict, problem_data = generate_robust_optimization_socp(n_vars=4, n_constraints=3)
    print(f"✓ Generated robust optimization problem with {problem_dict['n_variables']} variables")
    print(f"  SOCP constraints: {problem_dict['n_soc_constraints']}")
    print(f"  Uncertainty bounds: {[f'{b:.3f}' for b in problem_dict['uncertainty_bounds']]}")
    
    # Test solving with different solvers
    solvers_to_test = ["CLARABEL", "SCS", "ECOS"]  # OSQP doesn't support general SOCP well
    
    for solver in solvers_to_test:
        try:
            if hasattr(cp, solver):
                results = solve_robust_optimization(solver)
                status = results["status"]
                print(f"  {solver}: {status}")
                
                if results.get("optimal_value") is not None:
                    print(f"    Optimal value: {results['optimal_value']:.6f}")
                    print(f"    Robust cost: {results['robust_cost']:.3f}")
                    if results.get("solve_time"):
                        print(f"    Solve time: {results['solve_time']:.3f}s")
            else:
                print(f"  {solver}: Not available")
                
        except Exception as e:
            print(f"  {solver}: Error - {e}")
    
    print("\n✓ Robust optimization SOCP test completed!")