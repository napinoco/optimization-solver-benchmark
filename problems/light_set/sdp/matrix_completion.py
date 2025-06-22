"""
Matrix Completion SDP Problem
=============================

Matrix completion with nuclear norm minimization formulated as SDP.

Problem formulation:
    minimize    trace(W1) + trace(W2)
    subject to  [W1   X ]  ⪰ 0      [PSD constraint]
                [X^T  W2]
                X_ij = M_ij  for (i,j) ∈ Ω    [Known entries constraint]

where:
- X: matrix to be completed (decision variable)
- W1, W2: auxiliary matrices for nuclear norm relaxation
- M: partially observed matrix
- Ω: set of observed entries

This is a classic SDP relaxation of the matrix completion problem,
where nuclear norm minimization promotes low-rank solutions.
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


def generate_matrix_completion_sdp(n_rows: int = 4, n_cols: int = 4, observation_rate: float = 0.6, seed: int = 42) -> Tuple[Dict[str, Any], ProblemData]:
    """
    Generate a matrix completion SDP problem.
    
    Args:
        n_rows: Number of rows in the matrix
        n_cols: Number of columns in the matrix
        observation_rate: Fraction of entries that are observed
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problem_dict, ProblemData)
    """
    np.random.seed(seed)
    
    # Generate a low-rank ground truth matrix
    rank = min(2, min(n_rows, n_cols) - 1)
    U = np.random.randn(n_rows, rank)
    V = np.random.randn(n_cols, rank)
    true_matrix = U @ V.T
    
    # Create observation mask
    n_observations = int(n_rows * n_cols * observation_rate)
    all_indices = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    observed_indices = np.random.choice(len(all_indices), n_observations, replace=False)
    
    # Create partially observed matrix
    observed_matrix = np.full((n_rows, n_cols), np.nan)
    observation_mask = np.zeros((n_rows, n_cols), dtype=bool)
    
    for idx in observed_indices:
        i, j = all_indices[idx]
        observed_matrix[i, j] = true_matrix[i, j]
        observation_mask[i, j] = True
    
    # Add small amount of noise to observed entries
    noise_level = 0.01
    for i in range(n_rows):
        for j in range(n_cols):
            if observation_mask[i, j]:
                observed_matrix[i, j] += noise_level * np.random.randn()
    
    # Decision variables
    X = cp.Variable((n_rows, n_cols), name="completed_matrix")  # Matrix to complete
    W1 = cp.Variable((n_rows, n_rows), name="nuclear_norm_aux1")  # Auxiliary for nuclear norm
    W2 = cp.Variable((n_cols, n_cols), name="nuclear_norm_aux2")  # Auxiliary for nuclear norm
    
    # Objective: minimize nuclear norm (via trace minimization)
    objective = cp.Minimize(cp.trace(W1) + cp.trace(W2))
    
    # Constraints
    constraints = []
    
    # SDP constraint: Nuclear norm relaxation
    # [W1   X ]  ⪰ 0
    # [X^T  W2]
    top_block = cp.hstack([W1, X])
    bottom_block = cp.hstack([X.T, W2])
    sdp_matrix = cp.vstack([top_block, bottom_block])
    constraints.append(sdp_matrix >> 0)  # PSD constraint
    
    # Observation constraints: X_ij = M_ij for observed entries
    for i in range(n_rows):
        for j in range(n_cols):
            if observation_mask[i, j]:
                constraints.append(X[i, j] == observed_matrix[i, j])
    
    # Create CVXPY problem
    problem = cp.Problem(objective, constraints)
    
    # Create problem data structure
    problem_dict = {
        "name": "matrix_completion",
        "description": "Matrix completion with nuclear norm minimization via SDP",
        "problem_class": "SDP",
        "n_variables": n_rows * n_cols + n_rows * n_rows + n_cols * n_cols,  # X + W1 + W2
        "n_constraints": len(constraints),
        "n_sdp_constraints": 1,
        "matrix_dimensions": {
            "target_matrix": (n_rows, n_cols),
            "sdp_matrix": (n_rows + n_cols, n_rows + n_cols),
            "aux_matrices": [(n_rows, n_rows), (n_cols, n_cols)]
        },
        "objective_sense": "minimize",
        "true_matrix": true_matrix.tolist(),
        "observed_matrix": observed_matrix.tolist(),
        "observation_mask": observation_mask.tolist(),
        "observation_rate": observation_rate,
        "true_rank": rank,
        "cvxpy_problem": problem,
        "variables": {"X": X, "W1": W1, "W2": W2}
    }
    
    # Create ProblemData instance
    problem_data = ProblemData(
        name="matrix_completion",
        problem_class="SDP",
        cvxpy_problem=problem,
        variables={"X": X, "W1": W1, "W2": W2},
        objective=objective,
        constraints=constraints,
        metadata={
            "description": "Matrix completion with nuclear norm minimization via SDP",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_sdp_constraints": 1,
            "observation_rate": observation_rate,
            "true_rank": rank,
            "n_observations": n_observations,
            "matrix_dimensions": {
                "target": (n_rows, n_cols),
                "sdp": (n_rows + n_cols, n_rows + n_cols)
            },
            "ground_truth": {
                "matrix": true_matrix,
                "rank": rank
            },
            "observations": {
                "matrix": observed_matrix,
                "mask": observation_mask,
                "count": n_observations
            }
        }
    )
    
    return problem_dict, problem_data


def solve_matrix_completion(solver_name: str = "CLARABEL") -> Dict[str, Any]:
    """Solve the matrix completion problem and return results."""
    
    problem_dict, problem_data = generate_matrix_completion_sdp()
    problem = problem_dict["cvxpy_problem"]
    
    # Solve the problem
    try:
        problem.solve(solver=getattr(cp, solver_name), verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            X_opt = problem_dict["variables"]["X"].value
            W1_opt = problem_dict["variables"]["W1"].value
            W2_opt = problem_dict["variables"]["W2"].value
            
            # Calculate completion metrics
            true_matrix = np.array(problem_dict["true_matrix"])
            observation_mask = np.array(problem_dict["observation_mask"])
            
            # Reconstruction error on unobserved entries
            unobserved_mask = ~observation_mask
            if np.any(unobserved_mask):
                reconstruction_error = np.linalg.norm(
                    (X_opt - true_matrix)[unobserved_mask]
                ) / np.sum(unobserved_mask)
            else:
                reconstruction_error = 0.0
            
            # Nuclear norm of solution
            nuclear_norm = np.sum(np.linalg.svd(X_opt, compute_uv=False))
            
            # Rank of solution (approximate)
            svd_values = np.linalg.svd(X_opt, compute_uv=False)
            effective_rank = np.sum(svd_values > 1e-6 * svd_values[0])
            
            results = {
                "status": problem.status,
                "optimal_value": problem.value,
                "completed_matrix": X_opt.tolist(),
                "reconstruction_error": float(reconstruction_error),
                "nuclear_norm": float(nuclear_norm),
                "effective_rank": int(effective_rank),
                "true_rank": problem_dict["true_rank"],
                "singular_values": svd_values.tolist(),
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
    # Test the matrix completion problem
    print("Testing Matrix Completion SDP Problem...")
    
    # Generate problem
    problem_dict, problem_data = generate_matrix_completion_sdp(n_rows=4, n_cols=4)
    print(f"✓ Generated matrix completion problem with {problem_dict['n_variables']} variables")
    print(f"  Target matrix: {problem_dict['matrix_dimensions']['target_matrix']}")
    print(f"  SDP matrix: {problem_dict['matrix_dimensions']['sdp_matrix']}")
    print(f"  SDP constraints: {problem_dict['n_sdp_constraints']}")
    print(f"  Observation rate: {problem_dict['observation_rate']:.1%}")
    print(f"  True rank: {problem_dict['true_rank']}")
    
    # Test solving with different solvers
    solvers_to_test = ["CLARABEL", "SCS"]  # Only these support SDP
    
    for solver in solvers_to_test:
        try:
            if hasattr(cp, solver):
                print(f"\n--- Testing {solver} ---")
                results = solve_matrix_completion(solver)
                status = results["status"]
                print(f"  Status: {status}")
                
                if results.get("optimal_value") is not None:
                    print(f"  Optimal value: {results['optimal_value']:.6f}")
                    print(f"  Reconstruction error: {results['reconstruction_error']:.6f}")
                    print(f"  Nuclear norm: {results['nuclear_norm']:.3f}")
                    print(f"  Effective rank: {results['effective_rank']} (true: {results['true_rank']})")
                    if results.get("solve_time"):
                        print(f"  Solve time: {results['solve_time']:.3f}s")
                    
                    # Show singular values
                    sv = results["singular_values"][:5]  # First 5
                    print(f"  Top singular values: {[f'{s:.3f}' for s in sv]}")
                else:
                    print(f"  Error: {results.get('error', 'Unknown error')}")
            else:
                print(f"\n--- {solver} ---")
                print(f"  Not available")
                
        except Exception as e:
            print(f"\n--- {solver} ---")
            print(f"  Error: {e}")
    
    print("\n✓ Matrix completion SDP test completed!")