"""
Portfolio Optimization SOCP Problem
====================================

Markowitz mean-variance portfolio optimization formulated as Second-Order Cone Program.

Problem formulation:
    minimize    t + gamma * s
    subject to  ||Sigma^(1/2) * x||_2 <= t        [SOCP constraint]
                ||x||_2 <= s                      [SOCP constraint]  
                mu^T * x >= r_min                 [Expected return constraint]
                sum(x) = 1                        [Budget constraint]
                x >= 0                            [Long-only constraint]

where:
- x: portfolio weights (decision variables)
- mu: expected returns vector
- Sigma: covariance matrix of returns
- r_min: minimum required return
- gamma: penalty parameter for norm regularization
- t, s: auxiliary variables for SOCP constraints

This is a classic SOCP formulation combining risk minimization with return constraints.
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


def generate_portfolio_optimization_socp(n_assets: int = 5, seed: int = 42) -> Tuple[Dict[str, Any], ProblemData]:
    """
    Generate a portfolio optimization SOCP problem.
    
    Args:
        n_assets: Number of assets in the portfolio
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problem_dict, ProblemData)
    """
    np.random.seed(seed)
    
    # Generate expected returns (higher risk -> higher expected return)
    mu = 0.05 + 0.05 * np.random.rand(n_assets)
    
    # Generate covariance matrix (positive semidefinite)
    A = np.random.randn(n_assets, n_assets)
    Sigma = A.T @ A
    Sigma_sqrt = np.linalg.cholesky(Sigma)
    
    # Problem parameters
    r_min = 0.08  # Minimum required return (8%)
    gamma = 0.1   # Regularization parameter
    
    # Decision variables
    x = cp.Variable(n_assets, name="portfolio_weights")  # Portfolio weights
    t = cp.Variable(name="risk_measure")                 # Risk auxiliary variable
    s = cp.Variable(name="norm_regularization")          # Norm regularization variable
    
    # Objective: minimize risk + regularization
    objective = cp.Minimize(t + gamma * s)
    
    # Constraints
    constraints = [
        # SOCP constraint: portfolio risk <= t
        cp.SOC(t, Sigma_sqrt.T @ x),
        
        # SOCP constraint: portfolio norm <= s
        cp.SOC(s, x),
        
        # Expected return constraint
        mu.T @ x >= r_min,
        
        # Budget constraint (fully invested)
        cp.sum(x) == 1,
        
        # Long-only constraint
        x >= 0
    ]
    
    # Create CVXPY problem
    problem = cp.Problem(objective, constraints)
    
    # Create problem data structure
    problem_dict = {
        "name": "portfolio_optimization",
        "description": "Markowitz portfolio optimization as SOCP",
        "problem_class": "SOCP",
        "n_variables": n_assets + 2,  # x + t + s
        "n_constraints": len(constraints),
        "n_soc_constraints": 2,
        "objective_sense": "minimize",
        "expected_returns": mu.tolist(),
        "covariance_matrix": Sigma.tolist(),
        "min_return": r_min,
        "regularization": gamma,
        "cvxpy_problem": problem,
        "variables": {"x": x, "t": t, "s": s}
    }
    
    # Create ProblemData instance
    problem_data = ProblemData(
        name="portfolio_optimization",
        problem_class="SOCP",
        cvxpy_problem=problem,
        variables={"x": x, "t": t, "s": s},
        objective=objective,
        constraints=constraints,
        metadata={
            "description": "Markowitz portfolio optimization as SOCP",
            "n_assets": n_assets,
            "n_soc_constraints": 2,
            "expected_returns": mu,
            "covariance_matrix": Sigma,
            "min_return": r_min,
            "regularization": gamma
        }
    )
    
    return problem_dict, problem_data


def solve_portfolio_optimization(solver_name: str = "CLARABEL") -> Dict[str, Any]:
    """Solve the portfolio optimization problem and return results."""
    
    problem_dict, problem_data = generate_portfolio_optimization_socp()
    problem = problem_dict["cvxpy_problem"]
    
    # Solve the problem
    try:
        problem.solve(solver=getattr(cp, solver_name), verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            x_opt = problem_dict["variables"]["x"].value
            t_opt = problem_dict["variables"]["t"].value
            s_opt = problem_dict["variables"]["s"].value
            
            # Calculate portfolio metrics
            mu = np.array(problem_dict["expected_returns"])
            Sigma = np.array(problem_dict["covariance_matrix"])
            
            portfolio_return = mu.T @ x_opt
            portfolio_risk = np.sqrt(x_opt.T @ Sigma @ x_opt)
            
            results = {
                "status": problem.status,
                "optimal_value": problem.value,
                "portfolio_weights": x_opt.tolist(),
                "portfolio_return": float(portfolio_return),
                "portfolio_risk": float(portfolio_risk),
                "risk_measure": float(t_opt),
                "norm_regularization": float(s_opt),
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
    # Test the portfolio optimization problem
    print("Testing Portfolio Optimization SOCP Problem...")
    
    # Generate problem
    problem_dict, problem_data = generate_portfolio_optimization_socp(n_assets=5)
    print(f"✓ Generated portfolio problem with {problem_dict['n_variables']} variables")
    print(f"  SOCP constraints: {problem_dict['n_soc_constraints']}")
    print(f"  Expected returns: {[f'{r:.3f}' for r in problem_dict['expected_returns']]}")
    
    # Test solving with different solvers
    solvers_to_test = ["CLARABEL", "SCS", "ECOS", "OSQP"]
    
    for solver in solvers_to_test:
        try:
            if hasattr(cp, solver):
                results = solve_portfolio_optimization(solver)
                status = results["status"]
                print(f"  {solver}: {status}")
                
                if results.get("optimal_value") is not None:
                    print(f"    Optimal value: {results['optimal_value']:.6f}")
                    print(f"    Portfolio return: {results['portfolio_return']:.3f}")
                    print(f"    Portfolio risk: {results['portfolio_risk']:.3f}")
                    if results.get("solve_time"):
                        print(f"    Solve time: {results['solve_time']:.3f}s")
            else:
                print(f"  {solver}: Not available")
                
        except Exception as e:
            print(f"  {solver}: Error - {e}")
    
    print("\n✓ Portfolio optimization SOCP test completed!")