# Configuration Guide

This guide explains how to configure the optimization solver benchmark system, including setup for MATLAB solvers.

## Overview

The benchmark system uses YAML configuration files to manage solvers, problems, and system settings. All configuration files are located in the `config/` directory.

## Configuration Files

### 1. Problem Registry (`config/problem_registry.yaml`)

Defines available benchmark problems from external libraries (DIMACS, SDPLIB): file paths, formats, and known objective values.

### 2. Site Configuration (`config/site_config.yaml`)

Contains site metadata for the generated reports and the solver `display_order`.

**Note**: Solver definitions are not configured via YAML. Python solver configurations live in `scripts/solvers/python/solver_configs.py` (single source of truth), and MATLAB solver configurations in `scripts/solvers/matlab/matlab_process_interface.py`. Availability is detected dynamically at runtime.

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

Working MATLAB integration reports both MATLAB solvers as available, e.g.:
```
MATLAB Solver Status: 2/2 working
  ✓ matlab_sedumi: SeDuMi
  ✓ matlab_sdpt3: SDPT3
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

For debugging MATLAB integration issues, see `scripts/solvers/matlab/matlab_process_interface.py` and check `logs/benchmark.log` (the process interface logs MATLAB availability detection at startup).

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

See [Development Guidelines](../development/detail_design.md#development-guidelines) in detail_design.md — the entry point is `scripts/solvers/python/solver_configs.py`.

## Related Documentation

- **[Local Development Guide](LOCAL_DEVELOPMENT_GUIDE.md)**: Setting up development environment
- **[External Libraries Guide](EXTERNAL_LIBRARIES.md)**: Problem library configuration
- **[GitHub Actions Setup](GITHUB_ACTIONS_SETUP.md)**: CI/CD configuration
- **[Export Guide](EXPORT_GUIDE.md)**: Data export and reporting configuration

---

*This configuration guide covers all aspects of system setup including MATLAB solver integration. For additional help, refer to the troubleshooting section or consult the other guides in this directory.*