# External Library Setup Guide

This guide explains how to set up and use external optimization problem libraries (DIMACS and SDPLIB) with the benchmark system.

## Overview

The benchmark system supports external problem libraries to test solvers on real-world optimization problems:

- **DIMACS**: SeDuMi `.mat` format problems
- **SDPLIB**: SDPA `.dat-s` format semidefinite programming problems
- **Automatic Integration**: Problems are automatically detected and converted to CVXPY format

Exact problem counts change over time; `config/problem_registry.yaml` is the authoritative list (see Verify Setup below for how to count them yourself).

## Quick Setup

### 1. Clone External Libraries

The external libraries are included as Git submodules. They should already be available if you cloned the repository with `--recursive`:

```bash
# Check if libraries are present
ls problems/DIMACS/
ls problems/SDPLIB/

# If missing, clone manually (matches .gitmodules)
git submodule update --init problems/DIMACS problems/SDPLIB
```

### 2. Verify Setup

```bash
# Validate external libraries are properly detected
python main.py --validate

# Count registered problems per library
python -c "
import yaml
with open('config/problem_registry.yaml') as f:
    registry = yaml.safe_load(f)['problem_libraries']
for lib in ('DIMACS', 'SDPLIB'):
    count = sum(1 for p in registry.values() if p['library_name'] == lib)
    print(f'{lib} problems: {count}')
"
```

## Running External Library Benchmarks

### Basic Usage

```bash
# Run benchmarks on all external libraries
python main.py --benchmark --library_names DIMACS,SDPLIB

# Run only DIMACS problems
python main.py --benchmark --library_names DIMACS

# Run only SDPLIB problems
python main.py --benchmark --library_names SDPLIB

# Full benchmark with reporting
python main.py --all --library_names DIMACS,SDPLIB
```

### Advanced Usage

```bash
# Test specific solvers on external problems
python main.py --benchmark --solvers cvxpy_clarabel,cvxpy_scs --library_names DIMACS,SDPLIB

# Run with verbose output
python main.py --benchmark --library_names DIMACS,SDPLIB --verbose
```

## Understanding External Problem Results

External problems are significantly more challenging than synthetic test cases, and success rates vary by problem structure (LP/QP/SOCP/SDP) and solver. CLARABEL and SCS (via CVXPY) currently have the broadest compatibility across both libraries.

### Solver Compatibility

| Problem Type | CLARABEL | SCS | ECOS | OSQP | SciPy |
|-------------|----------|-----|------|------|-------|
| DIMACS (Mixed) | ✅ | ✅ | ❌* | ❌* | ❌ |
| SDPLIB (SDP) | ✅ | ✅ | ❌ | ❌ | ❌ |

*Limited support for specific problem types

## Configuration

### Problem Registry Structure

External library problems are registered as flat entries in `config/problem_registry.yaml` under `problem_libraries:`. See [Problem Registry Structure](../development/detail_design.md#problem-registry-structure) in detail_design.md for the exact schema (`display_name`, `file_path`, `file_type`, `library_name`, `for_test_flag`).

## Analyzing Results

External library results appear in all generated reports (see [detail_design.md](../development/detail_design.md#report-generation)) and in the JSON/CSV exports (see [EXPORT_GUIDE.md](EXPORT_GUIDE.md)), tagged with their `library_name`.

## Troubleshooting

### Common Issues

**Problem: "SDP problem must include a pre-built CVXPY problem"**
- Solution: This indicates the CVXPY converter is working correctly - the problem requires CVXPY conversion
- Status: Expected behavior for external libraries

**Problem: Libraries not found**
```bash
# Re-clone missing submodules
git submodule update --init problems/DIMACS problems/SDPLIB
```

**Problem: Low success rates**
- Expected: External problems are challenging real-world instances
- CLARABEL and SCS have the best compatibility
- Many problems may be infeasible or require specialized solvers

### Performance Tips

1. **Use CLARABEL**: Best overall performance on external problems
2. **Increase Timeouts**: Some problems require >300 seconds
3. **Filter by Library**: Use `--library_names` for focused testing
4. **Expect Failures**: Many external problems are challenging by design

## Adding New External Libraries

See [Adding New Problem Formats](../development/detail_design.md#adding-new-problem-formats) in detail_design.md for the current loader architecture and registration steps.

## References

### External Library Sources
- **DIMACS**: [DIMACS Implementation Challenges](http://dimacs.rutgers.edu/)
- **SDPLIB**: [Semidefinite Programming Library](http://www.cise.ufl.edu/research/sparse/matrices/SDPLIB/)
- **VSDP Project**: [Verified SDP Library](https://github.com/vsdp)

### Related Documentation
- **[Technical Design](../development/detail_design.md)**: Problem loaders and data model
- **[Local Development Guide](LOCAL_DEVELOPMENT_GUIDE.md)**: Running benchmarks locally
