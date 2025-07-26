# Optimization Solver Benchmark System

A research tool for benchmarking optimization solvers across LP, QP, SOCP, and SDP problems using external problem libraries (DIMACS, SDPLIB) with automated execution and HTML report generation.

Benchmark results are continuously published at: https://napinoco.github.io/optimization-solver-benchmark/

> **⚠️ Development Status**: This system is currently under active development. Some solvers and problem classes have not yet been properly measured, and comprehensive benchmarking coverage across all problem types is still being completed.

## Overview

**Purpose**: Compare optimization solver performance using standardized problem sets and minimal solver configuration for unbiased evaluation.

**Problem Libraries**:
- **DIMACS**: 47 problems in SeDuMi .mat format
- **SDPLIB**: 92+ problems in SDPA .dat-s format  
- **Total**: 139+ optimization problems

**Supported Solvers**:
- **Python (9)**: SciPy, CVXPY backends (CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HIGHS)
- **MATLAB/Octave (2)**: SeDuMi, SDPT3

## Quick Start

### Installation
```bash
git clone --recursive https://github.com/napinoco/optimization-solver-benchmark.git
cd optimization-solver-benchmark
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# For MATLAB/Octave solvers (optional)
# Ubuntu/Debian: sudo apt-get install octave
# macOS: brew install octave

# Setup MATLAB solvers - compiles MEX files for SeDuMi and SDPT3
matlab -batch "setup_matlab_solvers"  # or use octave --eval "setup_matlab_solvers"
```

### Usage
```bash
# Validate environment
python main.py --validate

# Run all benchmarks
python main.py --all

# Specific libraries
python main.py --benchmark --problem-set dimacs
python main.py --benchmark --problem-set sdplib

# Generate reports only
python main.py --report

# Test mode (no database changes)
python main.py --benchmark --dry-run

# Timeout configuration
python main.py --all --timeout 300                    # 5-minute timeout
python main.py --benchmark --timeout 60               # Quick tests with 1-minute limit
python main.py --benchmark --library_names SDPLIB --timeout 1800  # 30-minute timeout for difficult SDP problems
```

## Design Philosophy

**Fair Baseline Benchmarking**: Uses solver default parameters to avoid optimization bias and provide genuine "out of the box" performance comparison.

**Reproducible Results**: Problem libraries are pinned to specific commit hashes via git submodules, and solver versions are fixed in requirements.txt to ensure identical benchmarking environments across all executions.

**Complete Version Tracking**: All benchmark results are stored in a SQLite database with complete version information (solver versions, Git commit hashes, environment details) to enable historical analysis and ensure full reproducibility of any reported result.

## Project Structure

```
├── main.py                     # Entry point
├── config/                     # YAML configuration files
├── scripts/
│   ├── benchmark/              # Execution engine
│   ├── solvers/                # Python/MATLAB solver interfaces
│   ├── data_loaders/           # Problem format loaders (MAT/DAT)
│   ├── database/               # SQLite storage
│   └── reporting/              # HTML generation
├── problems/
│   ├── DIMACS/                 # External library (git submodule)
│   └── SDPLIB/                 # External library (git submodule)
├── database/                   # SQLite database files
│   └── results.db              # Benchmark results storage
└── docs/                       # Generated reports and documentation
```

## Architecture

**Modular Design**:
- **Loaders**: Convert problem files → standardized ProblemData
- **Runners**: Execute solvers on ProblemData → SolverResult  
- **Interfaces**: Coordinate problem loading and solver execution

**Multi-Language Support**: Unified Python/MATLAB integration via JSON data exchange and standardized result format.

## Adding Components

### New Solvers
**Python**: Add configuration to `PYTHON_SOLVER_CONFIGS` in `python_interface.py`  
**MATLAB**: Create `{solver}_runner.m` and add to `MATLAB_SOLVER_CONFIGS` in `matlab_interface.py`

### New Problems
Add external libraries as git submodules in `problems/` and extend loaders in `scripts/data_loaders/`

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Development context and instructions
- **[docs/development/](docs/development/)**: Technical design and implementation details

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **External Problem Libraries**: VSDP team for DIMACS and SDPLIB repositories
- **MATLAB Solvers**: SQLP team for maintaining Git repositories of SeDuMi and SDPT3 solvers  
- **Open-Source Solvers**: SciPy, CVXPY, CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HIGHS communities
- **Development Support**: Claude Code for coding assistance and development guidance