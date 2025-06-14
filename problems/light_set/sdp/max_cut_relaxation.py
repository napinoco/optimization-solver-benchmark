"""
Max Cut SDP Relaxation Problem
==============================

Semidefinite relaxation of the Maximum Cut problem on a graph.

Problem formulation:
    maximize    (1/4) * sum_{(i,j) in E} w_ij * (1 - X_ij)
    subject to  X ⪰ 0                   [X is positive semidefinite]
                X_ii = 1    ∀i           [Diagonal entries are 1]

where:
- X: n×n matrix variable with X_ij representing correlation between vertices i and j
- w_ij: edge weights in the graph
- E: set of edges

This is the famous Goemans-Williamson SDP relaxation of Max Cut,
which achieves a 0.878-approximation of the optimal integer solution.
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


def generate_max_cut_sdp(n_vertices: int = 5, edge_probability: float = 0.6, seed: int = 42) -> Tuple[Dict[str, Any], ProblemData]:
    """
    Generate a Max Cut SDP relaxation problem.
    
    Args:
        n_vertices: Number of vertices in the graph
        edge_probability: Probability of edge existence (for random graph)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problem_dict, ProblemData)
    """
    np.random.seed(seed)
    
    # Generate random graph adjacency matrix
    adjacency = np.zeros((n_vertices, n_vertices))
    edge_weights = {}
    edges = []
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if np.random.rand() < edge_probability:
                # Random edge weight
                weight = np.random.uniform(0.5, 2.0)
                adjacency[i, j] = weight
                adjacency[j, i] = weight  # Symmetric
                edge_weights[(i, j)] = weight
                edges.append((i, j))
    
    # Create Laplacian matrix for Max Cut
    # L_ij = w_ij if (i,j) is an edge, L_ii = -sum_j L_ij
    laplacian = np.zeros((n_vertices, n_vertices))
    for i in range(n_vertices):
        for j in range(n_vertices):
            if i != j and adjacency[i, j] > 0:
                laplacian[i, j] = -adjacency[i, j]
            elif i == j:
                laplacian[i, i] = np.sum(adjacency[i, :])
    
    # Decision variable: Correlation matrix X
    X = cp.Variable((n_vertices, n_vertices), name="correlation_matrix", symmetric=True)
    
    # Objective: maximize (1/4) * sum_{(i,j) in E} w_ij * (1 - X_ij)
    # Equivalent to: maximize (1/4) * trace(L) - (1/4) * trace(L * X)
    # Since trace(L) is constant, this becomes: minimize trace(L * X)
    objective = cp.Minimize(cp.trace(laplacian @ X))
    
    # Constraints
    constraints = []
    
    # X ⪰ 0 (X is positive semidefinite)
    constraints.append(X >> 0)
    
    # X_ii = 1 for all i (diagonal constraints)
    for i in range(n_vertices):
        constraints.append(X[i, i] == 1)
    
    # Create CVXPY problem
    problem = cp.Problem(objective, constraints)
    
    # Calculate some graph properties
    total_weight = np.sum([weight for weight in edge_weights.values()])
    n_edges = len(edges)
    
    # Create problem data structure
    problem_dict = {
        "name": "max_cut_relaxation",
        "description": "Goemans-Williamson SDP relaxation of Max Cut",
        "problem_class": "SDP",
        "n_variables": n_vertices * (n_vertices + 1) // 2,  # Symmetric matrix
        "n_constraints": len(constraints),
        "n_sdp_constraints": 1,
        "matrix_dimensions": {
            "correlation_matrix": (n_vertices, n_vertices),
            "graph_adjacency": (n_vertices, n_vertices)
        },
        "objective_sense": "minimize",  # Minimizing -objective to maximize
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "total_edge_weight": total_weight,
        "edge_probability": edge_probability,
        "adjacency_matrix": adjacency.tolist(),
        "laplacian_matrix": laplacian.tolist(),
        "edges": edges,
        "edge_weights": edge_weights,
        "cvxpy_problem": problem,
        "variables": {"X": X}
    }
    
    # Create ProblemData instance
    problem_data = ProblemData(
        name="max_cut_relaxation",
        problem_class="SDP",
        cvxpy_problem=problem,
        variables={"X": X},
        objective=objective,
        constraints=constraints,
        metadata={
            "description": "Goemans-Williamson SDP relaxation of Max Cut",
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "n_sdp_constraints": 1,
            "edge_probability": edge_probability,
            "total_weight": total_weight,
            "graph_properties": {
                "adjacency": adjacency,
                "laplacian": laplacian,
                "edges": edges,
                "edge_weights": edge_weights
            },
            "matrix_dimensions": {
                "correlation": (n_vertices, n_vertices),
                "graph": (n_vertices, n_vertices)
            }
        }
    )
    
    return problem_dict, problem_data


def solve_max_cut_relaxation(solver_name: str = "CLARABEL") -> Dict[str, Any]:
    """Solve the Max Cut SDP relaxation and return results."""
    
    problem_dict, problem_data = generate_max_cut_sdp()
    problem = problem_dict["cvxpy_problem"]
    
    # Solve the problem
    try:
        problem.solve(solver=getattr(cp, solver_name), verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            X_opt = problem_dict["variables"]["X"].value
            
            # Calculate SDP relaxation value
            laplacian = np.array(problem_dict["laplacian_matrix"])
            sdp_value = -problem.value  # Convert back to maximization
            
            # Analyze the solution matrix
            X_eigenvals = np.linalg.eigvals(X_opt)
            min_eigenval = np.min(X_eigenvals)
            is_psd = min_eigenval > -1e-8
            
            # Check diagonal constraints
            diagonal_values = np.diag(X_opt)
            diagonal_satisfied = np.allclose(diagonal_values, 1.0, atol=1e-6)
            
            # Rank of solution (indicates tightness of relaxation)
            rank = np.sum(X_eigenvals > 1e-8 * np.max(X_eigenvals))
            
            # Simple rounding heuristic for integer solution
            # Random hyperplane rounding (simplified)
            np.random.seed(42)  # For reproducible rounding
            try:
                # Cholesky decomposition for rounding
                L = np.linalg.cholesky(X_opt + 1e-8 * np.eye(len(X_opt)))
                random_vector = np.random.randn(len(X_opt))
                cut_vector = np.sign(L @ random_vector)
                
                # Calculate cut value
                adjacency = np.array(problem_dict["adjacency_matrix"])
                cut_value = 0.0
                for i in range(len(cut_vector)):
                    for j in range(i + 1, len(cut_vector)):
                        if adjacency[i, j] > 0 and cut_vector[i] != cut_vector[j]:
                            cut_value += adjacency[i, j]
                
                approximation_ratio = cut_value / sdp_value if sdp_value > 0 else 0
            except np.linalg.LinAlgError:
                cut_value = None
                approximation_ratio = None
            
            results = {
                "status": problem.status,
                "optimal_value": problem.value,
                "sdp_relaxation_value": float(sdp_value),
                "correlation_matrix": X_opt.tolist(),
                "correlation_eigenvalues": X_eigenvals.tolist(),
                "min_eigenvalue": float(min_eigenval),
                "is_positive_semidefinite": bool(is_psd),
                "diagonal_satisfied": bool(diagonal_satisfied),
                "solution_rank": int(rank),
                "rounded_cut_value": float(cut_value) if cut_value is not None else None,
                "approximation_ratio": float(approximation_ratio) if approximation_ratio is not None else None,
                "graph_properties": {
                    "n_vertices": problem_dict["n_vertices"],
                    "n_edges": problem_dict["n_edges"],
                    "total_weight": problem_dict["total_edge_weight"]
                },
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
    # Test the Max Cut SDP relaxation
    print("Testing Max Cut SDP Relaxation Problem...")
    
    # Generate problem
    problem_dict, problem_data = generate_max_cut_sdp(n_vertices=5)
    print(f"✓ Generated Max Cut problem with {problem_dict['n_variables']} variables")
    print(f"  Graph: {problem_dict['n_vertices']} vertices, {problem_dict['n_edges']} edges")
    print(f"  Correlation matrix: {problem_dict['matrix_dimensions']['correlation_matrix']}")
    print(f"  SDP constraints: {problem_dict['n_sdp_constraints']}")
    print(f"  Total edge weight: {problem_dict['total_edge_weight']:.3f}")
    print(f"  Edge probability: {problem_dict['edge_probability']:.1%}")
    
    # Test solving with different solvers
    solvers_to_test = ["CLARABEL", "SCS"]  # Only these support SDP
    
    for solver in solvers_to_test:
        try:
            if hasattr(cp, solver):
                print(f"\n--- Testing {solver} ---")
                results = solve_max_cut_relaxation(solver)
                status = results["status"]
                print(f"  Status: {status}")
                
                if results.get("optimal_value") is not None:
                    print(f"  SDP relaxation value: {results['sdp_relaxation_value']:.6f}")
                    print(f"  Min eigenvalue: {results['min_eigenvalue']:.6f}")
                    print(f"  Is PSD: {results['is_positive_semidefinite']}")
                    print(f"  Diagonal satisfied: {results['diagonal_satisfied']}")
                    print(f"  Solution rank: {results['solution_rank']}")
                    
                    if results['rounded_cut_value'] is not None:
                        print(f"  Rounded cut value: {results['rounded_cut_value']:.3f}")
                        print(f"  Approximation ratio: {results['approximation_ratio']:.3f}")
                    
                    if results.get("solve_time"):
                        print(f"  Solve time: {results['solve_time']:.3f}s")
                else:
                    print(f"  Error: {results.get('error', 'Unknown error')}")
            else:
                print(f"\n--- {solver} ---")
                print(f"  Not available")
                
        except Exception as e:
            print(f"\n--- {solver} ---")
            print(f"  Error: {e}")
    
    print("\n✓ Max Cut SDP relaxation test completed!")