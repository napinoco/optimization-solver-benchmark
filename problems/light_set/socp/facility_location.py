"""
Facility Location SOCP Problem
===============================

Capacitated facility location with distance minimization formulated as SOCP.

Problem formulation:
    minimize    sum_i sum_j (c_ij * x_ij) + sum_i (f_i * y_i)
    subject to  ||p_i - q_j||_2 <= t_ij     [SOCP constraint: Euclidean distance]
                t_ij >= M * x_ij             [Distance activation constraint]
                sum_j x_ij <= cap_i * y_i    [Capacity constraint]
                sum_i x_ij = demand_j        [Demand satisfaction]
                x_ij >= 0, y_i ∈ {0,1}      [Non-negativity and binary]

where:
- x_ij: amount shipped from facility i to customer j
- y_i: binary variable (1 if facility i is opened)
- t_ij: distance from facility i to customer j
- p_i: location of facility i
- q_j: location of customer j
- c_ij: unit shipping cost
- f_i: fixed opening cost for facility i
- cap_i: capacity of facility i
- demand_j: demand of customer j

This SOCP relaxation uses continuous variables instead of binary for y_i.
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


def generate_facility_location_socp(n_facilities: int = 3, n_customers: int = 5, seed: int = 42) -> Tuple[Dict[str, Any], ProblemData]:
    """
    Generate a facility location SOCP problem.
    
    Args:
        n_facilities: Number of potential facility locations
        n_customers: Number of customers
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problem_dict, ProblemData)
    """
    np.random.seed(seed)
    
    # Generate random facility locations (2D coordinates)
    facility_locations = np.random.rand(n_facilities, 2) * 10
    
    # Generate random customer locations (2D coordinates)
    customer_locations = np.random.rand(n_customers, 2) * 10
    
    # Generate facility parameters
    fixed_costs = 50 + 50 * np.random.rand(n_facilities)  # Fixed opening costs
    capacities = 10 + 20 * np.random.rand(n_facilities)   # Facility capacities
    
    # Generate customer demands
    demands = 2 + 3 * np.random.rand(n_customers)
    
    # Generate unit shipping costs (distance-based with noise)
    shipping_costs = np.zeros((n_facilities, n_customers))
    for i in range(n_facilities):
        for j in range(n_customers):
            distance = np.linalg.norm(facility_locations[i] - customer_locations[j])
            shipping_costs[i, j] = 1.0 + 0.1 * distance + 0.2 * np.random.rand()
    
    # Decision variables
    x = cp.Variable((n_facilities, n_customers), name="shipment")  # Shipment amounts
    y = cp.Variable(n_facilities, name="facility_open")             # Facility opening (relaxed to continuous [0,1])
    t = cp.Variable((n_facilities, n_customers), name="distances")  # Distance variables
    
    # Objective: minimize total cost (shipping + fixed costs)
    shipping_cost = cp.sum(cp.multiply(shipping_costs, x))
    fixed_cost = fixed_costs.T @ y
    objective = cp.Minimize(shipping_cost + fixed_cost)
    
    # Constraints
    constraints = []
    
    # SOCP constraints: Euclidean distance constraints
    for i in range(n_facilities):
        for j in range(n_customers):
            # ||facility_i - customer_j||_2 <= t_ij
            diff = facility_locations[i] - customer_locations[j]
            constraints.append(cp.SOC(t[i, j], diff))
    
    # Distance activation constraints: t_ij >= distance_base * x_ij
    # This links distance to shipment (if we ship, we pay the distance cost)
    M = 20.0  # Big-M parameter
    for i in range(n_facilities):
        for j in range(n_customers):
            constraints.append(t[i, j] >= 0.1 * x[i, j])  # Minimum distance cost per unit
    
    # Capacity constraints: total shipment from facility i <= capacity_i * y_i
    for i in range(n_facilities):
        constraints.append(cp.sum(x[i, :]) <= capacities[i] * y[i])
    
    # Demand satisfaction: sum of shipments to customer j = demand_j
    for j in range(n_customers):
        constraints.append(cp.sum(x[:, j]) == demands[j])
    
    # Variable bounds
    constraints.append(x >= 0)                    # Non-negative shipments
    constraints.append(y >= 0)                    # Relaxed binary variables
    constraints.append(y <= 1)                    # Upper bound for facility opening
    constraints.append(t >= 0)                    # Non-negative distances
    
    # Create CVXPY problem
    problem = cp.Problem(objective, constraints)
    
    # Create problem data structure
    problem_dict = {
        "name": "facility_location",
        "description": "Capacitated facility location with SOCP distance constraints",
        "problem_class": "SOCP",
        "n_variables": n_facilities * n_customers + n_facilities + n_facilities * n_customers,  # x + y + t
        "n_constraints": len(constraints),
        "n_soc_constraints": n_facilities * n_customers,
        "objective_sense": "minimize",
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        "facility_locations": facility_locations.tolist(),
        "customer_locations": customer_locations.tolist(),
        "fixed_costs": fixed_costs.tolist(),
        "capacities": capacities.tolist(),
        "demands": demands.tolist(),
        "shipping_costs": shipping_costs.tolist(),
        "cvxpy_problem": problem,
        "variables": {"x": x, "y": y, "t": t}
    }
    
    # Create ProblemData instance
    problem_data = ProblemData(
        name="facility_location",
        problem_class="SOCP",
        cvxpy_problem=problem,
        variables={"x": x, "y": y, "t": t},
        objective=objective,
        constraints=constraints,
        metadata={
            "description": "Capacitated facility location with SOCP distance constraints",
            "n_facilities": n_facilities,
            "n_customers": n_customers,
            "n_soc_constraints": n_facilities * n_customers,
            "total_demand": np.sum(demands),
            "total_capacity": np.sum(capacities),
            "facility_data": {
                "locations": facility_locations,
                "fixed_costs": fixed_costs,
                "capacities": capacities
            },
            "customer_data": {
                "locations": customer_locations,
                "demands": demands
            },
            "cost_structure": {
                "shipping_costs": shipping_costs
            }
        }
    )
    
    return problem_dict, problem_data


def solve_facility_location(solver_name: str = "CLARABEL") -> Dict[str, Any]:
    """Solve the facility location problem and return results."""
    
    problem_dict, problem_data = generate_facility_location_socp()
    problem = problem_dict["cvxpy_problem"]
    
    # Solve the problem
    try:
        problem.solve(solver=getattr(cp, solver_name), verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            x_opt = problem_dict["variables"]["x"].value
            y_opt = problem_dict["variables"]["y"].value
            t_opt = problem_dict["variables"]["t"].value
            
            # Calculate solution metrics
            total_cost = problem.value
            
            # Determine which facilities are "open" (y > threshold)
            facility_threshold = 0.1
            open_facilities = y_opt > facility_threshold
            n_open_facilities = np.sum(open_facilities)
            
            # Calculate utilization
            capacities = np.array(problem_dict["capacities"])
            utilization = np.sum(x_opt, axis=1) / capacities
            avg_utilization = np.mean(utilization[open_facilities]) if n_open_facilities > 0 else 0
            
            results = {
                "status": problem.status,
                "optimal_value": problem.value,
                "total_cost": float(total_cost),
                "n_open_facilities": int(n_open_facilities),
                "open_facilities": open_facilities.tolist(),
                "facility_opening_levels": y_opt.tolist(),
                "avg_facility_utilization": float(avg_utilization),
                "shipment_matrix": x_opt.tolist(),
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
    # Test the facility location problem
    print("Testing Facility Location SOCP Problem...")
    
    # Generate problem
    problem_dict, problem_data = generate_facility_location_socp(n_facilities=3, n_customers=5)
    print(f"✓ Generated facility location problem with {problem_dict['n_variables']} variables")
    print(f"  Facilities: {problem_dict['n_facilities']}, Customers: {problem_dict['n_customers']}")
    print(f"  SOCP constraints: {problem_dict['n_soc_constraints']}")
    print(f"  Total demand: {np.sum(problem_dict['demands']):.2f}")
    print(f"  Total capacity: {np.sum(problem_dict['capacities']):.2f}")
    
    # Test solving with different solvers
    solvers_to_test = ["CLARABEL", "SCS", "ECOS"]  # OSQP has limited SOCP support
    
    for solver in solvers_to_test:
        try:
            if hasattr(cp, solver):
                results = solve_facility_location(solver)
                status = results["status"]
                print(f"  {solver}: {status}")
                
                if results.get("optimal_value") is not None:
                    print(f"    Total cost: {results['total_cost']:.2f}")
                    print(f"    Open facilities: {results['n_open_facilities']}")
                    print(f"    Avg utilization: {results['avg_facility_utilization']:.2f}")
                    if results.get("solve_time"):
                        print(f"    Solve time: {results['solve_time']:.3f}s")
            else:
                print(f"  {solver}: Not available")
                
        except Exception as e:
            print(f"  {solver}: Error - {e}")
    
    print("\n✓ Facility location SOCP test completed!")