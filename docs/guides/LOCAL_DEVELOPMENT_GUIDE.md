# Local Development Guide

This guide walks you through setting up, running, and contributing to the optimization solver benchmark system on your local machine.

## Table of Contents
- [Setup](#setup)
- [Understanding the Output](#understanding-the-output)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Contributing](#contributing)

---

## Setup

For installation, cloning with submodules, and running your first benchmark, see the [README Quick Start](../../README.md#quick-start).

### Alternative Python Environment Setup

**Using pyenv**
```bash
pyenv install 3.12.2
pyenv local 3.12.2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda**
```bash
conda create -n solver-benchmark python=3.12
conda activate solver-benchmark
```

---

## Understanding the Output

### Generated Files

**HTML Reports** (`docs/pages/`)
- `index.html` - Main dashboard with overview statistics
- `results_matrix.html` - Problems × solvers results matrix
- `raw_data.html` - Raw results table

**Data Files** (`docs/pages/data/`) - see [EXPORT_GUIDE.md](EXPORT_GUIDE.md) for the full format reference and analysis examples

**Database** (`database/`)
- `results.db` - SQLite database with all results

### Key Metrics

**Success Rate**: Percentage of problems solved optimally
**Solve Time**: Wall-clock time for solver execution
**Problem Types**: LP (Linear), QP (Quadratic), SOCP (Second-Order Cone), SDP (Semidefinite)
**Solver Status**: see the full status code table in [detail_design.md](../development/detail_design.md#solverresult-scriptssolverssolver_interfacepy)

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

For the project structure, see [detail_design.md](../development/detail_design.md#directory-layout).

### Local Development Cycle

1. **Make Changes** to code in `scripts/`
2. **Test Changes** with `python main.py --validate`
3. **Run Benchmarks** with `python main.py --benchmark`
4. **Generate Reports** with `python main.py --report`
5. **Review Output** in `docs/pages/index.html`
6. **Commit Changes** including updated reports

### Adding New Solvers or Problems

See the [Development Guidelines](../development/detail_design.md#development-guidelines) in detail_design.md for the exact steps and files to touch.

---

## Testing

The test suite uses pytest (`tests/unit/` for component tests, `tests/integration/` for registry integrity checks). Configuration is in `pyproject.toml`.

```bash
# Install dev tools (not part of requirements.txt)
pip install pytest ruff

# Run the test suite
pytest tests/

# Lint and format checks (same as CI)
ruff check scripts/ main.py tests/
ruff format --check scripts/ main.py tests/
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
ls problems/DIMACS/data/
ls problems/SDPLIB/data/
cat config/problem_registry.yaml
```

### Performance Issues

**Slow Benchmark Execution**
- Check system resources (CPU, memory)
- Use fewer problems: `python main.py --all --library_names DIMACS` (instead of both DIMACS,SDPLIB)
- Test with fewer solvers: `--solvers scipy_linprog`

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

**Check Available Solvers**
```bash
python -c "
from scripts.benchmark.runner import BenchmarkRunner
runner = BenchmarkRunner()
print('Available solvers:', runner.get_available_solvers())
"
```

**Verify Problem Registry**
```bash
python -c "
from scripts.data_loaders.problem_loader import load_problem_registry
registry = load_problem_registry()
print(f'Total problems: {len(registry[\"problem_libraries\"])}')
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
3. **Check Generated Reports**: Verify `docs/pages/index.html` looks correct
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

*This guide covers the essential aspects of local development. For technical details, see [detail_design.md](../development/detail_design.md). For project overview and installation, see the main [README.md](../../README.md).*

*Last Updated: July 2026*
