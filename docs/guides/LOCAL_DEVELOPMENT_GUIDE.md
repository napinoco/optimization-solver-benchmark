# Local Development Guide

This guide walks you through setting up, running, and contributing to the optimization solver benchmark system on your local machine.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running Benchmarks](#running-benchmarks)
- [Understanding the Output](#understanding-the-output)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
- Python 3.9+ 
- Git
- 4GB+ RAM
- 1GB+ free disk space

### 30-Second Setup
```bash
# Clone the repository
git clone https://github.com/your-org/optimization-solver-benchmark.git
cd optimization-solver-benchmark

# Install dependencies
pip install -r requirements/base.txt -r requirements/python.txt

# Validate environment
python main.py --validate

# Run complete benchmark and generate reports
python main.py --all
```

### View Results
Open `docs/index.html` in your web browser to see the generated reports.

---

## Installation

### 1. Python Environment Setup

**Option A: Using pyenv (Recommended)**
```bash
# Install Python 3.12
pyenv install 3.12.2
pyenv local 3.12.2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: Using conda**
```bash
conda create -n solver-benchmark python=3.12
conda activate solver-benchmark
```

### 2. Install Dependencies

**Core Dependencies**
```bash
pip install -r requirements/base.txt
```

**Python Solver Dependencies**
```bash
pip install -r requirements/python.txt
```

**Optional: Export Dependencies (for advanced reporting)**
```bash
pip install -r requirements/export.txt
```

### 3. Verify Installation
```bash
python main.py --validate
```

Expected output:
```
✅ Environment validation passed
✅ Configuration loaded successfully
✅ 5 solvers available
✅ 8 problems loaded from light_set
```

---

## Running Benchmarks

### Basic Commands

**Environment Check**
```bash
python main.py --validate
```

**Run Benchmarks Only**
```bash
python main.py --benchmark
```

**Generate Reports Only** (from existing results)
```bash
python main.py --report
```

**Complete Workflow** (benchmark + reports)
```bash
python main.py --all
```

### Advanced Options

**Verbose Output**
```bash
python main.py --all --verbose
```

**Specific Solvers**
```bash
python main.py --all --solvers scipy,clarabel_cvxpy
```

**Custom Configuration**
```bash
python main.py --all --config config/custom_config.yaml
```

**Quiet Mode**
```bash
python main.py --all --quiet
```

### Command Reference

| Command | Description |
|---------|-------------|
| `--validate` | Check environment setup |
| `--benchmark` | Run benchmarks only |
| `--report` | Generate reports only |
| `--all` | Run benchmarks + generate reports |
| `--verbose` | Enable detailed logging |
| `--quiet` | Suppress non-error output |
| `--solvers` | Specify comma-separated solver list |
| `--config` | Use custom configuration file |

---

## Understanding the Output

### Generated Files

**HTML Reports** (`docs/`)
- `index.html` - Main dashboard with overview statistics
- `solver_comparison.html` - Detailed solver performance comparison
- `problem_analysis.html` - Problem-specific analysis
- `results_matrix.html` - Interactive results matrix
- `statistical_analysis.html` - Statistical analysis and trends
- `performance_profiling.html` - Performance profiling report
- `environment_info.html` - System and environment information

**Data Files** (`docs/data/`)
- `results.json` - Complete benchmark results (programmatic access)
- `results.csv` - Results in CSV format (spreadsheet analysis)
- `summary.json` - Summary statistics and solver comparison
- `metadata.json` - Environment and configuration metadata
- `statistical_analysis_report.json` - Statistical analysis data
- `performance_profiling_report.json` - Performance profiling data

**Database** (`database/`)
- `results.db` - SQLite database with all results

### Key Metrics

**Success Rate**: Percentage of problems solved optimally
**Solve Time**: Wall-clock time for solver execution
**Problem Types**: LP (Linear), QP (Quadratic), SOCP (Second-Order Cone), SDP (Semidefinite)
**Solver Status**: optimal, error, timeout, infeasible, unbounded

### Interpreting Results

**Good Performance Indicators:**
- High success rate (>90%)
- Low and consistent solve times
- Successful solving across multiple problem types

**Potential Issues:**
- Low success rate (<80%) - may indicate solver compatibility issues
- High variance in solve times - may indicate numerical instability
- Frequent "error" status - check solver installation

---

## Development Workflow

### Project Structure
```
optimization-solver-benchmark/
├── main.py                    # Main entry point
├── config/                    # Configuration files
├── scripts/                   # Core implementation
│   ├── benchmark/            # Benchmark execution
│   ├── solvers/              # Solver implementations
│   ├── reporting/            # Report generation
│   └── utils/                # Utilities
├── problems/light_set/       # Test problems
├── docs/                     # Generated reports (committed)
├── database/                 # SQLite database
└── tests/                    # Test suite
```

### Local Development Cycle

1. **Make Changes** to code in `scripts/`
2. **Test Changes** with `python main.py --validate`
3. **Run Benchmarks** with `python main.py --benchmark`
4. **Generate Reports** with `python main.py --report`
5. **Review Output** in `docs/index.html`
6. **Commit Changes** including updated reports

### Adding New Solvers

1. **Create Solver Class** in `scripts/solvers/python/`
```python
from scripts.solvers.solver_interface import SolverInterface, SolverResult

class NewSolver(SolverInterface):
    def solve(self, problem: ProblemData) -> SolverResult:
        # Implementation here
        pass
```

2. **Add Configuration** in `config/solvers.yaml`
```yaml
new_solver:
  name: "New Solver"
  environment: "python"
  enabled: true
  timeout: 300
```

3. **Add Dependencies** in `requirements/python.txt`
```
new-solver-package>=1.0.0
```

4. **Test Integration**
```bash
python main.py --validate
python main.py --benchmark --solvers new_solver
```

### Adding New Problems

1. **Add Problem Files** to `problems/light_set/[type]/`
2. **Update Registry** in `problems/problem_registry.yaml`
```yaml
problems:
  light_set:
    LP:
      - name: "new_problem"
        file_path: "problems/light_set/lp/new_problem.mps"
        problem_class: "LP"
```

3. **Test Loading**
```bash
python -c "from scripts.data_loaders.problem_loader import load_problem; print(load_problem('new_problem', 'light_set'))"
```

---

## Testing

### Test Structure
```
tests/
├── unit/          # Unit tests for individual components
├── integration/   # Integration tests for workflows  
├── debug/         # Debug and troubleshooting scripts
└── fixtures/      # Test data and fixtures
```

### Running Tests

**Individual Test Files**
```bash
python tests/integration/test_validation.py
python tests/debug/debug_integration.py
```

**Problem Loading Test**
```bash
python scripts/benchmark/problem_loader.py
```

**Solver Interface Test**
```bash
python scripts/benchmark/solver_interface.py
```

### Debug Scripts

**Check Python Integration**
```bash
python tests/debug/debug_integration.py
```

**Test Simple Integration**
```bash
python tests/integration/test_simple_integration.py
```

---

## Troubleshooting

### Common Issues

**"No module named 'cvxpy'"**
```bash
# Install CVXPY
pip install cvxpy>=1.3.0
```

**"Backend CLARABEL not available"**
```bash
# Install specific solver
pip install clarabel>=0.5.0
```

**"Permission denied" on database**
```bash
# Check database permissions
chmod 644 database/results.db
```

**"No problems found in registry"**
```bash
# Verify problem files exist
ls problems/light_set/lp/
cat problems/problem_registry.yaml
```

### Performance Issues

**Slow Benchmark Execution**
- Check system resources (CPU, memory)
- Reduce timeout in `config/benchmark_config.yaml`
- Test with fewer solvers: `--solvers scipy`

**Large Database Size**
```bash
# Check database size
sqlite3 database/results.db "SELECT COUNT(*) FROM results;"

# Clean old results if needed
sqlite3 database/results.db "DELETE FROM results WHERE benchmark_id < X;"
```

### Getting Help

**Environment Information**
```bash
python main.py --validate
python -c "import sys; print(sys.version)"
pip list | grep -E "(cvxpy|scipy|numpy)"
```

**Log Analysis**
```bash
# Check recent logs
tail -f logs/benchmark.log

# Search for errors
grep ERROR logs/benchmark.log
```

**Debug Mode**
```bash
# Run with maximum verbosity
python main.py --all --verbose
```

### Configuration Debugging

**Check Configuration Loading**
```bash
python -c "
from scripts.utils.config_loader import ConfigLoader
config = ConfigLoader('config/benchmark_config.yaml')
print(config.data)
"
```

**Verify Solver Configuration**
```bash
python -c "
from scripts.utils.config_loader import ConfigLoader
config = ConfigLoader('config/solvers.yaml')
for name, solver in config.data['solvers'].items():
    print(f'{name}: enabled={solver.get(\"enabled\", True)}')
"
```

---

## Best Practices

### Development Guidelines

1. **Test Early and Often**: Run `--validate` after any changes
2. **Commit Generated Files**: Include updated reports in your commits
3. **Use Descriptive Commit Messages**: Explain what benchmark changes you made
4. **Check Performance Impact**: Compare solve times before/after changes
5. **Document Configuration Changes**: Update this guide for new config options

### Performance Guidelines

1. **Fair Comparison**: Keep `parallel_jobs: 1` for consistent benchmarking
2. **Reasonable Timeouts**: Use 300s timeout for most problems
3. **Clean Environment**: Run in fresh virtual environment for reproducible results
4. **Monitor Resources**: Ensure adequate RAM and CPU for solver execution

### Git Workflow

```bash
# Standard development workflow
git checkout -b feature/new-solver
# Make changes
python main.py --all  # Generate updated reports
git add -A
git commit -m "Add new solver with benchmark results"
git push origin feature/new-solver
# Create pull request
```

---

## Contributing

### Before Submitting Pull Requests

1. **Validate Environment**: `python main.py --validate`
2. **Run Complete Benchmark**: `python main.py --all`
3. **Check Generated Reports**: Verify `docs/index.html` looks correct
4. **Run Tests**: Execute relevant test files
5. **Update Documentation**: Update this guide if you changed workflows

### Pull Request Checklist

- [ ] Environment validation passes
- [ ] All solvers execute successfully  
- [ ] Reports generate without errors
- [ ] Database contains expected results
- [ ] Tests pass
- [ ] Documentation updated (if needed)
- [ ] Performance impact assessed

---

*This guide covers the essential aspects of local development. For technical details, see [detail_design.md](detail_design.md). For project overview, see the main [README.md](../../README.md).*

*Last Updated: June 2025*