"""
Control Theory LMI (Linear Matrix Inequality) SDP Problem
==========================================================

Lyapunov stability analysis for linear dynamical systems via SDP.

Problem formulation:
    find        P
    subject to  P ≻ 0                    [P is positive definite]
                A^T P + P A ≺ 0          [Lyapunov inequality]

where:
- A: system matrix of linear dynamical system ẋ = Ax
- P: Lyapunov matrix (positive definite)

This SDP checks if the system ẋ = Ax is asymptotically stable.
If feasible, P serves as a Lyapunov function V(x) = x^T P x.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData


def generate_control_lmi_sdp(n_states: int = 3, stability_margin: float = 0.1, seed: int = 42) -> Tuple[Dict[str, Any], ProblemData]:
    """
    Generate a control LMI SDP problem for stability analysis.
    
    Args:
        n_states: Number of states in the dynamical system
        stability_margin: Margin for stability (larger = more stable)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problem_dict, ProblemData)
    """
    np.random.seed(seed)
    
    # Generate a stable system matrix A
    # Create random matrix and make it stable
    A_base = np.random.randn(n_states, n_states)
    
    # Make eigenvalues have negative real parts (stable)
    eigenvals, eigenvecs = np.linalg.eig(A_base)
    # Shift eigenvalues to be stable with margin
    stable_eigenvals = -stability_margin - np.abs(np.real(eigenvals)) - 1j * np.imag(eigenvals)
    
    # Reconstruct matrix with stable eigenvalues
    A = np.real(eigenvecs @ np.diag(stable_eigenvals) @ np.linalg.inv(eigenvecs))
    
    # Decision variable: Lyapunov matrix P
    P = cp.Variable((n_states, n_states), name="lyapunov_matrix", symmetric=True)
    
    # Objective: minimize trace (for well-conditioning)
    objective = cp.Minimize(cp.trace(P))
    
    # Constraints
    constraints = []
    
    # P ≻ 0 (P is positive definite)
    constraints.append(P >> 0)
    
    # Lyapunov inequality: A^T P + P A ≺ 0
    lyapunov_lmi = A.T @ P + P @ A
    constraints.append(lyapunov_lmi << 0)
    
    # Normalization constraint to avoid trivial solution P = 0
    constraints.append(cp.trace(P) >= 1)
    
    # Create CVXPY problem
    problem = cp.Problem(objective, constraints)
    
    # Check theoretical stability (eigenvalues of A)
    A_eigenvals = np.linalg.eigvals(A)
    max_real_eigenval = np.max(np.real(A_eigenvals))
    is_theoretically_stable = max_real_eigenval < 0
    
    # Create problem data structure
    problem_dict = {
        "name": "control_lmi",
        "description": "Lyapunov stability analysis via LMI",
        "problem_class": "SDP",
        "n_variables": n_states * (n_states + 1) // 2,  # Symmetric matrix variables
        "n_constraints": len(constraints),
        "n_sdp_constraints": 2,  # P >> 0 and -LMI >> 0
        "matrix_dimensions": {
            "system_matrix": (n_states, n_states),
            "lyapunov_matrix": (n_states, n_states)
        },
        "objective_sense": "minimize",
        "system_matrix": A.tolist(),
        "system_eigenvalues": A_eigenvals.tolist(),
        "max_real_eigenvalue": float(max_real_eigenval),
        "is_theoretically_stable": is_theoretically_stable,
        "stability_margin": stability_margin,
        "cvxpy_problem": problem,
        "variables": {"P": P}
    }
    
    # Create ProblemData instance
    problem_data = ProblemData(
        name="control_lmi",
        problem_class="SDP",
        cvxpy_problem=problem,
        variables={"P": P},
        objective=objective,
        constraints=constraints,
        metadata={
            "description": "Lyapunov stability analysis via LMI",
            "n_states": n_states,
            "n_sdp_constraints": 2,
            "stability_margin": stability_margin,
            "system_properties": {
                "matrix": A,
                "eigenvalues": A_eigenvals,
                "max_real_eigenvalue": max_real_eigenval,
                "theoretically_stable": is_theoretically_stable
            },
            "matrix_dimensions": {
                "system": (n_states, n_states),
                "lyapunov": (n_states, n_states)
            }
        }
    )
    
    return problem_dict, problem_data


def solve_control_lmi(solver_name: str = "CLARABEL") -> Dict[str, Any]:
    """Solve the control LMI problem and return results."""
    
    problem_dict, problem_data = generate_control_lmi_sdp()
    problem = problem_dict["cvxpy_problem"]
    
    # Solve the problem
    try:
        problem.solve(solver=getattr(cp, solver_name), verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            P_opt = problem_dict["variables"]["P"].value
            A = np.array(problem_dict["system_matrix"])
            
            # Verify the solution
            # Check if P is positive definite
            P_eigenvals = np.linalg.eigvals(P_opt)
            min_P_eigenval = np.min(P_eigenvals)
            is_P_positive_definite = min_P_eigenval > 1e-8
            
            # Check Lyapunov inequality A^T P + P A < 0
            lyapunov_matrix = A.T @ P_opt + P_opt @ A
            lyap_eigenvals = np.linalg.eigvals(lyapunov_matrix)
            max_lyap_eigenval = np.max(lyap_eigenvals)
            satisfies_lyapunov = max_lyap_eigenval < -1e-8
            
            # Condition number of P
            condition_number = np.linalg.cond(P_opt)
            
            results = {
                "status": problem.status,
                "optimal_value": problem.value,
                "lyapunov_matrix": P_opt.tolist(),
                "lyapunov_eigenvalues": P_eigenvals.tolist(),
                "min_lyapunov_eigenvalue": float(min_P_eigenval),
                "is_positive_definite": bool(is_P_positive_definite),
                "lyapunov_lmi_eigenvalues": lyap_eigenvals.tolist(),
                "max_lmi_eigenvalue": float(max_lyap_eigenval),
                "satisfies_lyapunov_inequality": bool(satisfies_lyapunov),
                "condition_number": float(condition_number),
                "system_eigenvalues": problem_dict["system_eigenvalues"],
                "max_real_system_eigenvalue": problem_dict["max_real_eigenvalue"],
                "theoretically_stable": problem_dict["is_theoretically_stable"],
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
    # Test the control LMI problem
    print("Testing Control LMI SDP Problem...")
    
    # Generate problem
    problem_dict, problem_data = generate_control_lmi_sdp(n_states=3)
    print(f"✓ Generated control LMI problem with {problem_dict['n_variables']} variables")
    print(f"  System matrix: {problem_dict['matrix_dimensions']['system_matrix']}")
    print(f"  Lyapunov matrix: {problem_dict['matrix_dimensions']['lyapunov_matrix']}")
    print(f"  SDP constraints: {problem_dict['n_sdp_constraints']}")
    print(f"  Stability margin: {problem_dict['stability_margin']}")
    
    # Show system eigenvalues
    eigenvals = problem_dict["system_eigenvalues"]
    print(f"  System eigenvalues: {[f'{np.real(e):.3f}{np.imag(e):+.3f}j' for e in eigenvals]}")
    print(f"  Max real eigenvalue: {problem_dict['max_real_eigenvalue']:.6f}")
    print(f"  Theoretically stable: {problem_dict['is_theoretically_stable']}")
    
    # Test solving with different solvers
    solvers_to_test = ["CLARABEL", "SCS"]  # Only these support SDP
    
    for solver in solvers_to_test:
        try:
            if hasattr(cp, solver):
                print(f"\n--- Testing {solver} ---")
                results = solve_control_lmi(solver)
                status = results["status"]
                print(f"  Status: {status}")
                
                if results.get("optimal_value") is not None:
                    print(f"  Optimal value: {results['optimal_value']:.6f}")
                    print(f"  Min Lyapunov eigenvalue: {results['min_lyapunov_eigenvalue']:.6f}")
                    print(f"  P is positive definite: {results['is_positive_definite']}")
                    print(f"  Max LMI eigenvalue: {results['max_lmi_eigenvalue']:.6f}")
                    print(f"  Satisfies Lyapunov ineq: {results['satisfies_lyapunov_inequality']}")
                    print(f"  Condition number: {results['condition_number']:.2f}")
                    if results.get("solve_time"):
                        print(f"  Solve time: {results['solve_time']:.3f}s")
                    
                    # Verification
                    stable = results['is_positive_definite'] and results['satisfies_lyapunov_inequality']
                    print(f"  ✓ Stability verified: {stable}")
                else:
                    print(f"  Error: {results.get('error', 'Unknown error')}")
            else:
                print(f"\n--- {solver} ---")
                print(f"  Not available")
                
        except Exception as e:
            print(f"\n--- {solver} ---")
            print(f"  Error: {e}")
    
    print("\n✓ Control LMI SDP test completed!")