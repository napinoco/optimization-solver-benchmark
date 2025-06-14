# Octave Integration Guide

The optimization solver benchmark system supports GNU Octave as a MATLAB-compatible optimization environment. This document explains how to set up and use Octave solvers.

## Overview

Octave integration provides:
- **MATLAB-compatible optimization** using GNU Octave
- **Linear programming** with `linprog` function
- **Quadratic programming** with `quadprog` function
- **Cross-platform support** (Linux, macOS, Windows)
- **Automatic installation detection** and configuration

## Installation

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install octave octave-optim
```

### macOS (Homebrew)
```bash
brew install octave
```

### Windows (Chocolatey)
```bash
choco install octave.portable
```

### Manual Installation
Download from [GNU Octave official website](https://www.gnu.org/software/octave/) and follow installation instructions for your platform.

## Verification

Test your Octave installation:
```bash
# Test basic Octave functionality
octave --eval "printf('Octave is working\\n');"

# Test optimization package
octave --eval "pkg load optim; printf('Optim package loaded\\n');"

# Run benchmark test
python scripts/solvers/octave/test_octave.py
```

## Configuration

### Automatic Configuration
The system automatically detects Octave installations in common locations:
- `/usr/bin/octave` (Linux)
- `/usr/local/bin/octave` (macOS)
- `/opt/homebrew/bin/octave` (macOS Homebrew)

### Manual Configuration
Specify Octave path in `config/octave_config.yaml`:
```yaml
octave:
  executable_path: "/path/to/your/octave"
```

Or in solver configuration (`config/solvers.yaml`):
```yaml
octave_linprog:
  octave_path: "/path/to/your/octave"
  enabled: true
```

## Available Solvers

### Octave LinProg
- **Problem Types**: Linear Programming (LP)
- **Algorithm**: Interior-point method
- **Configuration**: `octave_linprog` in solvers.yaml

### Octave QuadProg
- **Problem Types**: Quadratic Programming (QP)
- **Algorithm**: Interior-point method
- **Configuration**: `octave_quadprog` in solvers.yaml

## Usage Examples

### Command Line
```bash
# Run benchmark with Octave solvers
python scripts/benchmark/runner.py --solver octave_linprog --problem SIMPLE_LP

# Test specific Octave solver
python scripts/solvers/octave/test_octave.py
```

### Python API
```python
from scripts.solvers.octave.octave_runner import OctaveSolver
from scripts.benchmark.problem_loader import load_problem

# Create solver
solver = OctaveSolver(name="My Octave Solver")

# Load and solve problem
problem = load_problem("SIMPLE_LP")
result = solver.solve(problem)

print(f"Status: {result.status}")
print(f"Objective: {result.objective_value}")
```

## Problem Format

Octave solvers accept problems in standard optimization format:

### Linear Programming
```
minimize    c^T x
subject to  A_ineq x <= b_ineq
            A_eq x = b_eq
            lb <= x <= ub
```

### Quadratic Programming
```
minimize    (1/2) x^T Q x + c^T x
subject to  A_ineq x <= b_ineq
            A_eq x = b_eq
            lb <= x <= ub
```

## Performance Notes

- **Single-threaded**: Octave solvers run sequentially
- **Memory usage**: Generally lower than CVXPY backends
- **Speed**: Competitive for small-medium problems
- **Compatibility**: Excellent for MATLAB code validation

## Troubleshooting

### Common Issues

#### Octave Not Found
```
Error: Octave not found or not properly installed
```
**Solution**: Install Octave and ensure it's in your PATH

#### Missing Optimization Package
```
Error: optim package not available
```
**Solution**: Install optimization package:
```bash
octave --eval "pkg install -forge optim"
```

#### Permission Issues (Linux/macOS)
```
Error: Permission denied
```
**Solution**: Check Octave executable permissions:
```bash
chmod +x /usr/bin/octave
```

#### Windows Path Issues
```
Error: Could not execute Octave
```
**Solution**: Add Octave to Windows PATH or specify full path in configuration

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.getLogger("octave_solver").setLevel(logging.DEBUG)
```

## Performance Comparison

Typical performance characteristics compared to Python solvers:

| Aspect | Octave | Python (SciPy) | Python (CVXPY) |
|--------|--------|----------------|----------------|
| Setup Time | Medium | Fast | Medium |
| Small Problems | Fast | Fast | Medium |
| Large Problems | Medium | Fast | Fast |
| Memory Usage | Low | Medium | High |
| Debugging | Good | Excellent | Good |

## CI/CD Integration

The system includes GitHub Actions workflows for automated Octave testing:
- **Ubuntu**: Tests with `apt-get` installation
- **macOS**: Tests with Homebrew installation  
- **Windows**: Tests with Chocolatey installation

See `.github/workflows/octave_test.yml` for configuration details.

## Contributing

When adding new Octave solvers:

1. **Follow naming convention**: `octave_<algorithm>`
2. **Add to solvers.yaml**: Include configuration
3. **Update tests**: Add test cases in `test_octave.py`
4. **Document parameters**: Update this guide with new options

## Limitations

Current limitations of Octave integration:
- **Problem types**: LP and QP only (no SOCP/SDP yet)
- **Parallel execution**: Not supported
- **Advanced options**: Limited compared to commercial solvers
- **Platform dependencies**: Requires separate Octave installation

## Future Enhancements

Planned improvements:
- **SOCP support** via specialized packages
- **SDP support** with appropriate Octave packages
- **Parallel execution** where possible
- **Enhanced error handling** and diagnostics
- **Performance optimizations** for large problems