# Solution Storage

This directory contains optimal solutions from benchmark runs for verification and analysis purposes.

## Directory Structure

```
solutions/
├── dimacs/        # Solutions for DIMACS problems
├── sdplib/        # Solutions for SDPLIB problems
└── README.md      # This documentation
```

## File Format

Solutions are stored in NumPy compressed format (`.npz`) with the following structure:

```python
{
    'primal_solution': np.array,      # Primal variables (x)
    'dual_solution': np.array,        # Dual variables (y) 
    'primal_objective': float,        # Primal objective value
    'dual_objective': float,          # Dual objective value
    'duality_gap': float,             # |primal - dual|
    'primal_infeasibility': float,    # ||Ax - b|| / (1 + ||b||)
    'dual_infeasibility': float,      # ||A^T y - c|| / (1 + ||c||)
    'solver_name': str,               # Solver that found the solution
    'solve_time': float,              # Time to solve
    'timestamp': str                  # ISO timestamp
}
```

## Usage

Solutions are automatically saved when `--save-solutions` flag is used during benchmarking.

```bash
python main.py --benchmark --save-solutions
```

Solutions can be loaded for analysis:

```python
import numpy as np
solution = np.load('solutions/dimacs/nb_CLARABEL.npz')
x = solution['primal_solution']
y = solution['dual_solution']
```