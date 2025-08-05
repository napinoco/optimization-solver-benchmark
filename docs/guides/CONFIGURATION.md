# Configuration Guide

This guide explains how to configure the optimization solver benchmark system, including setup for MATLAB solvers.

## Overview

The benchmark system uses YAML configuration files to manage solvers, problems, and system settings. All configuration files are located in the `config/` directory.

## Configuration Files

### 1. Solver Registry (`config/solver_registry.yaml`)

The solver registry defines all available solvers and their display names for reports.

```yaml
solvers:
  # Python solvers (always available)
  scipy_linprog:
    display_name: "SciPy linprog"
  
  cvxpy_clarabel:
    display_name: "CLARABEL (via CVXPY)"
  
  cvxpy_scs:
    display_name: "SCS (via CVXPY)"
  
  cvxpy_ecos:
    display_name: "ECOS (via CVXPY)"
  
  cvxpy_osqp:
    display_name: "OSQP (via CVXPY)"
  
  cvxpy_cvxopt:
    display_name: "CVXOPT (via CVXPY)"
  
  cvxpy_sdpa:
    display_name: "SDPA (via CVXPY)"
  
  cvxpy_scip:
    display_name: "SCIP (via CVXPY)"
  
  cvxpy_highs:
    display_name: "HiGHS (via CVXPY)"
    
  # MATLAB solvers (require MATLAB installation)
  matlab_sedumi:
    display_name: "SeDuMi (via MATLAB)"
  
  matlab_sdpt3:
    display_name: "SDPT3 (via MATLAB)"
```

**Note**: The solver registry only contains display names. Actual solver initialization logic is implemented in code for better maintainability and dynamic availability detection.

### 2. Problem Registry (`config/problem_registry.yaml`)

Defines available benchmark problems from various libraries (DIMACS, SDPLIB, internal).

### 3. Site Configuration (`config/site_config.yaml`)

Contains general system settings and reporting configuration.

## MATLAB Solver Configuration

### Prerequisites

For MATLAB solvers to work, you need:

1. **MATLAB Installation**: MATLAB R2019b or later
2. **Solver Availability**: SeDuMi and SDPT3 must be installed and in MATLAB path
3. **Command Line Access**: `matlab` command must be available in system PATH

### MATLAB Solver Setup

#### 1. Install Required MATLAB Toolboxes

```matlab
% In MATLAB, verify SeDuMi is available:
help sedumi

% Verify SDPT3 is available:
help sdpt3
```

#### 2. Add Solvers to MATLAB Path

If solvers are not found, add them to your MATLAB path:

```matlab
% Add SeDuMi to path
addpath('/path/to/sedumi');

% Add SDPT3 to path  
addpath('/path/to/sdpt3');

% Save path
savepath;
```

#### 3. Verify Command Line Access

Test MATLAB command line execution:

```bash
# Test basic MATLAB execution
matlab -batch "disp('Hello from MATLAB')"

# Test SeDuMi availability
matlab -batch "help sedumi"

# Test SDPT3 availability  
matlab -batch "help sdpt3"
```

### Configuration Integration

#### Solver Availability Detection

The system automatically detects MATLAB solver availability:

- **Available**: If MATLAB is accessible and solvers are in path
- **Graceful Degradation**: If MATLAB is not available, only Python solvers are used
- **Dynamic Detection**: Availability is checked at runtime, not configuration time

#### Validation

Use the validation command to check MATLAB solver setup:

```bash
# Basic validation
python main.py --validate

# Detailed validation with solver status
python main.py --validate-verbose
```

Expected output for working MATLAB integration:
```
Solver Validation Results:
  Working Solvers: 11/11
  Working Problems: 139/139

MATLAB Solver Status: 2/2 working
  ✓ matlab_sedumi: SeDuMi (version 1.3.7)
  ✓ matlab_sdpt3: SDPT3 (version 4.0)
```

## Solver Filtering and Selection

### Command Line Usage

```bash
# Run specific solvers only
python main.py --benchmark --solvers cvxpy_clarabel,matlab_sedumi

# Run MATLAB solvers only
python main.py --benchmark --solvers matlab_sedumi,matlab_sdpt3

# Run Python solvers only  
python main.py --benchmark --solvers cvxpy_clarabel,cvxpy_scs,scipy_linprog
```

### Configuration-Based Filtering

Solvers can be filtered programmatically:

```python
from scripts.benchmark.runner import BenchmarkRunner

runner = BenchmarkRunner()
all_solvers = runner.get_available_solvers()

# Filter by type
matlab_solvers = [s for s in all_solvers if s.startswith('matlab_')]
python_solvers = [s for s in all_solvers if not s.startswith('matlab_')]

print(f"MATLAB solvers: {matlab_solvers}")
print(f"Python solvers: {python_solvers}")
```

## Troubleshooting

### Common Issues

#### 1. MATLAB Not Found

**Error**: `MATLAB solvers not available: matlab command not found`

**Solution**:
- Ensure MATLAB is installed and `matlab` command is in PATH
- On Windows: Add MATLAB bin directory to system PATH
- On macOS/Linux: Create symlink or add to PATH in shell profile

#### 2. SeDuMi/SDPT3 Not Found

**Error**: `SeDuMi solver initialization failed`

**Solution**:
- Install SeDuMi: Download from [SeDuMi website](http://sedumi.ie.lehigh.edu/)
- Install SDPT3: Download from [SDPT3 website](https://blog.nus.edu.sg/mattohkc/softwares/sdpt3/)
- Add to MATLAB path and save path permanently

#### 3. Permission Issues

**Error**: `Permission denied when executing MATLAB`

**Solution**:
- Check MATLAB license and user permissions
- Run benchmark system with appropriate user privileges
- Verify MATLAB can run in batch mode

#### 4. Version Compatibility

**Error**: `MATLAB version not supported`

**Solution**:
- Use MATLAB R2019b or later
- Update to latest MATLAB version if possible
- Check solver compatibility with MATLAB version

### Validation Commands

```bash
# Test environment setup
python main.py --validate

# Detailed solver information
python main.py --validate-verbose

# Test specific solver creation
python -c "
from scripts.benchmark.runner import BenchmarkRunner
runner = BenchmarkRunner()
sedumi = runner.create_solver('matlab_sedumi')
print(f'SeDuMi version: {sedumi.get_version()}')
"
```

### Debug Information

For debugging MATLAB integration issues:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test MATLAB solver
from scripts.solvers.matlab.matlab_interface import SeDuMiSolver
solver = SeDuMiSolver()
print(f"Solver initialized: {solver.solver_name}")
```

## Configuration Best Practices

### 1. Version Control

- Keep configuration files in version control
- Document any local modifications
- Use environment-specific configurations for different deployment targets

### 2. Solver Management

- Regularly validate solver availability
- Update display names to reflect actual solver versions
- Test configuration changes before deployment

### 3. MATLAB Environment

- Use consistent MATLAB versions across deployment environments
- Document required toolboxes and versions
- Automate MATLAB path setup in deployment scripts

### 4. Performance Considerations

- MATLAB solver initialization has ~6-second overhead per solver
- Consider pre-warming MATLAB in production environments
- Use solver filtering to avoid unnecessary MATLAB startup

## Advanced Configuration

### Custom Solver Addition

To add new solvers:

1. **Implement SolverInterface**: Create solver class inheriting from `SolverInterface`
2. **Update BenchmarkRunner**: Add solver creation logic to `create_solver()` method
3. **Add to Registry**: Include solver in `solver_registry.yaml`
4. **Update Documentation**: Document new solver requirements and setup

### Environment-Specific Settings

Create environment-specific configurations:

```yaml
# config/solver_registry_dev.yaml - Development environment
solvers:
  # Only fast solvers for development
  scipy_linprog:
    display_name: "SciPy linprog"
  cvxpy_clarabel:
    display_name: "CLARABEL (via CVXPY)"

# config/solver_registry_prod.yaml - Production environment  
solvers:
  # All solvers including MATLAB for comprehensive benchmarking
  # ... (full solver list)
```

## Related Documentation

- **[Local Development Guide](LOCAL_DEVELOPMENT_GUIDE.md)**: Setting up development environment
- **[External Libraries Guide](EXTERNAL_LIBRARIES.md)**: Problem library configuration
- **[GitHub Actions Setup](GITHUB_ACTIONS_SETUP.md)**: CI/CD configuration
- **[Export Guide](EXPORT_GUIDE.md)**: Data export and reporting configuration

---

*This configuration guide covers all aspects of system setup including MATLAB solver integration. For additional help, refer to the troubleshooting section or consult the other guides in this directory.*