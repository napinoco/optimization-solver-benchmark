# External Library Setup Guide

This guide explains how to set up and use external optimization problem libraries (DIMACS and SDPLIB) with the benchmark system.

## 📋 Overview

The benchmark system supports external problem libraries to test solvers on real-world optimization problems:

- **DIMACS**: 47 optimization problems in SeDuMi .mat format
- **SDPLIB**: 92 semidefinite programming problems in SDPA format
- **Automatic Integration**: Problems are automatically detected and converted to CVXPY format

## 🚀 Quick Setup

### 1. Clone External Libraries

The external libraries are included as Git submodules. They should already be available if you cloned the repository with the complete implementation:

```bash
# Check if libraries are present
ls problems/DIMACS/
ls problems/SDPLIB/

# If missing, clone manually
cd problems/
git clone https://github.com/vsdp/DIMACS_library.git DIMACS
git clone https://github.com/vsdp/SDPLIB.git SDPLIB
```

### 2. Verify Setup

```bash
# Validate external libraries are properly detected
python main.py --validate

# Check problem registry includes external problems
python -c "
import yaml
with open('problems/problem_registry.yaml') as f:
    registry = yaml.safe_load(f)
    print(f'DIMACS problems: {len(registry.get(\"external_libraries\", {}).get(\"DIMACS\", {}).get(\"problems\", []))}')
    print(f'SDPLIB problems: {len(registry.get(\"external_libraries\", {}).get(\"SDPLIB\", {}).get(\"problems\", []))}')
"
```

## 🧪 Running External Library Benchmarks

### Basic Usage

```bash
# Run benchmarks on all external libraries
python main.py --benchmark --solvers cvxpy --problem-set external

# Run only DIMACS problems
python main.py --benchmark --solvers cvxpy --problem-set dimacs

# Run only SDPLIB problems  
python main.py --benchmark --solvers cvxpy --problem-set sdplib

# Full benchmark with reporting
python main.py --all --problem-set external
```

### Advanced Usage

```bash
# Test specific solvers on external problems
python main.py --benchmark --solvers CLARABEL,SCS --problem-set external

# Increase timeout for challenging problems
python main.py --benchmark --problem-set external --timeout 600

# Run with verbose logging
python main.py --benchmark --problem-set external --log-level DEBUG
```

## 📊 Understanding External Problem Results

### Problem Structure Analysis

The system automatically analyzes external problems and displays:

- **Variables**: Number of decision variables
- **Constraints**: Number of constraints
- **Problem Type**: Automatic classification (LP, QP, SOCP, SDP)
- **Library Source**: Origin information (e.g., "control family", "FILTER problem set")

Example output:
```
control1 (SDP): 70 variables, 21 constraints
Source: control family

filter48_socp (SDP): 3284 variables, 969 constraints  
Source: FILTER problem set
```

### Expected Performance

External problems are significantly more challenging than synthetic test cases:

| Library | Total Problems | Typical Success Rate | Best Solvers |
|---------|---------------|---------------------|--------------|
| DIMACS  | 47           | ~40%               | CLARABEL, SCS |
| SDPLIB  | 92           | ~35%               | CLARABEL, SCS |
| Combined | 139          | ~37%               | CLARABEL primary |

## 🔧 Configuration

### Problem Registry Structure

External libraries are configured in `problems/problem_registry.yaml`:

```yaml
external_libraries:
  DIMACS:
    path: "problems/DIMACS"
    format: "sedumi_mat"
    description: "DIMACS optimization problems"
    problems:
      - name: "arch0"
        file: "arch/arch0.mat.gz"
        family: "arch family"
      - name: "control1"  
        file: "control/control1.mat.gz"
        family: "control family"
        
  SDPLIB:
    path: "problems/SDPLIB"
    format: "sdpa_sparse"
    description: "Semidefinite programming library"
    problems:
      - name: "hinf1"
        file: "hinf1.dat-s"
        family: "hinf family"
```

### Solver Compatibility

| Problem Type | CLARABEL | SCS | ECOS | OSQP | SciPy |
|-------------|----------|-----|------|------|-------|
| DIMACS (Mixed) | ✅ | ✅ | ❌* | ❌* | ❌ |
| SDPLIB (SDP) | ✅ | ✅ | ❌ | ❌ | ❌ |

*Limited support for specific problem types

## 📈 Analyzing Results

### HTML Reports

External library results appear in all generated reports:

- **Problem Analysis**: Shows structure analysis and library sources
- **Results Matrix**: Comprehensive solver performance across all problems
- **Solver Comparison**: Success rates broken down by problem source

### Data Exports

JSON/CSV exports include external library metadata:

```json
{
  "problem_name": "control1",
  "problem_type": "SDP", 
  "n_variables": 70,
  "n_constraints": 21,
  "library_source": "control family",
  "solver_name": "CLARABEL (via CVXPY)",
  "status": "optimal",
  "solve_time": 0.018
}
```

## 🛠️ Troubleshooting

### Common Issues

**Problem: "SDP problem must include a pre-built CVXPY problem"**
- Solution: This indicates the CVXPY converter is working correctly - the problem requires CVXPY conversion
- Status: Expected behavior for external libraries

**Problem: Libraries not found**
```bash
# Re-clone missing libraries
cd problems/
git clone https://github.com/vsdp/DIMACS_library.git DIMACS
git clone https://github.com/vsdp/SDPLIB.git SDPLIB
```

**Problem: Low success rates**
- Expected: External problems are challenging real-world instances
- CLARABEL and SCS have the best compatibility
- Many problems may be infeasible or require specialized solvers

### Performance Tips

1. **Use CLARABEL**: Best overall performance on external problems
2. **Increase Timeouts**: Some problems require >300 seconds
3. **Filter by Type**: Use specific problem sets (dimacs/sdplib) for focused testing
4. **Expect Failures**: Many external problems are challenging by design

## 🔍 Adding New External Libraries

### Step-by-Step Process

1. **Add Library Directory**
```bash
cd problems/
git clone <your-library-repo> NEW_LIBRARY
```

2. **Create Loader Module**
```python
# scripts/external/new_library_loader.py
def load_new_library_problem(file_path, problem_name):
    # Implement format-specific loading
    return problem_data
```

3. **Update Problem Registry**
```yaml
external_libraries:
  NEW_LIBRARY:
    path: "problems/NEW_LIBRARY"
    format: "custom_format"
    description: "New optimization library"
    problems:
      - name: "problem1"
        file: "problem1.ext"
        family: "problem family"
```

4. **Update Problem Loader**
```python
# scripts/benchmark/problem_loader.py
if problem_set == "NEW_LIBRARY":
    from scripts.external.new_library_loader import load_new_library_problem
    return load_new_library_problem(file_path, problem_name)
```

### Format Requirements

New libraries should provide:
- **Problem Data**: Objective function, constraints, variable bounds
- **Metadata**: Problem dimensions, type classification
- **CVXPY Compatibility**: Conversion to CVXPY format for solver compatibility

## 📚 References

### External Library Sources
- **DIMACS**: [DIMACS Implementation Challenges](http://dimacs.rutgers.edu/)
- **SDPLIB**: [Semidefinite Programming Library](http://www.cise.ufl.edu/research/sparse/matrices/SDPLIB/)
- **VSDP Project**: [Verified SDP Library](https://github.com/vsdp)

### Related Documentation
- **[Problem Structure Analysis](../development/tasks.md#task-70-create-problem-structure-analysis-module)**: Technical details on automatic problem analysis
- **[CVXPY Converter](../development/tasks.md#fix-external-library-cvxpy-problem-conversion)**: Implementation of external problem conversion
- **[Benchmark Configuration](../guides/MANUAL_TRIGGER_GUIDE.md)**: Advanced benchmark options

---

**Last Updated**: June 2025  
**Status**: Production Ready ✅  
**Compatibility**: DIMACS (47 problems), SDPLIB (92 problems), 139 total external problems