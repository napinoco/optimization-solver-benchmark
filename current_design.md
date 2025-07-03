# MATLAB Integration Current Design Documentation

> **Document Purpose**: This document explains the current functionalities and data-process-flow of MATLAB-related code components in the optimization solver benchmark system.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Flow Pipeline](#data-flow-pipeline)
4. [Integration Mechanisms](#integration-mechanisms)
5. [Problem Resolution System](#problem-resolution-system)
6. [Result Processing Chain](#result-processing-chain)
7. [Error Handling Strategy](#error-handling-strategy)
8. [Performance Considerations](#performance-considerations)

---

## Architecture Overview

The MATLAB integration provides a complete bridge between MATLAB optimization solvers (SeDuMi, SDPT3) and the Python-based benchmark system. The architecture follows a **pipeline design** with clear separation of concerns:

```
Python System ↔ matlab_solver.py ↔ MATLAB Scripts ↔ Solver Backends
```

### Key Design Principles

- **SolverInterface Compliance**: Full adherence to the Python `SolverInterface` specification
- **Problem Registry Integration**: Seamless resolution of DIMACS/SDPLIB problems via YAML configuration
- **Fault Tolerance**: Comprehensive error handling with meaningful fallbacks at every layer
- **Data Standardization**: Consistent JSON interchange format between Python and MATLAB
- **Temporary File Management**: Safe handling of intermediate files with automatic cleanup

---

## Core Components

### 1. Python Layer: `matlab_solver.py`

**Purpose**: Primary Python interface implementing `SolverInterface` for MATLAB solvers.

**Key Classes**:
- `MatlabSolver`: Base class with full SolverInterface compliance
- `SeDuMiSolver`: Convenience wrapper for SeDuMi
- `SDPT3Solver`: Convenience wrapper for SDPT3

**Core Functionality**:
- **Environment Verification**: Validates MATLAB/Octave availability and startup performance
- **Version Detection**: Dynamic detection and caching of MATLAB and solver versions
- **Problem Validation**: Registry-based problem compatibility checking
- **Subprocess Management**: Safe MATLAB execution with timeout handling
- **Temporary File Management**: UUID-based temporary files with automatic cleanup

### 2. MATLAB Orchestrator: `matlab_runner.m`

**Purpose**: Consolidated MATLAB entry point containing all utility functions and coordinating problem loading, solver execution, and result saving.

**Execution Flow**:
1. **Configuration Loading**: Uses integrated YAML reader for problem metadata
2. **Problem Loading**: Dispatches to appropriate loader based on file type
3. **Solver Execution**: Calls specific solver runner (SeDuMi/SDPT3)
4. **Metrics Calculation**: Uses integrated metrics calculator for objective values and feasibility measures
5. **Result Serialization**: Uses integrated JSON formatter and file saver

**Integrated Architecture**: All utility functions are embedded as nested functions within the main orchestrator, eliminating external dependencies and simplifying deployment to a single 1,035-line file.

### 3. Solver Implementations

#### `sedumi_runner.m`
- **Configuration**: Minimal solver options for fair benchmarking (`fid=0` for silent mode)
- **Validation**: Comprehensive input validation and cone structure normalization
- **Error Handling**: Graceful failure with structured error result creation
- **Version Detection**: Multi-stage SeDuMi version identification

#### `sdpt3_runner.m`
- **Configuration**: Default SDPT3 parameters with verbose output disabled
- **Preprocessing**: Input validation and problem structure verification
- **Result Mapping**: Translation of SDPT3 status codes to standard format
- **Performance Monitoring**: Solve time measurement and iteration counting

### 4. Integrated Utility Functions

All utility functions have been consolidated into `matlab_runner.m` as nested functions for simplified deployment and reduced dependencies:

#### Core Utilities (Integrated)
- **`calculate_solver_metrics()`**: Objective computation and feasibility measures
- **`read_problem_registry()`**: YAML parsing for problem configuration  
- **`format_result_to_json()`**: MATLAB-to-JSON conversion with type safety
- **`save_json_safely()`**: Robust file I/O with error recovery
- **`save_solutions_if_needed()`**: Optional solution vector persistence
- **`detect_versions()`**: Comprehensive environment and solver version detection

#### Supporting Functions (Integrated)
- **`calculate_dual_cone_violation()`**: Cone-specific dual infeasibility calculation
- **`proj_onto_soc()`**: Second-order cone projection
- **`convert_to_json_compatible()`**: Data type conversion for JSON serialization
- **`get_field_safe()`**: Safe struct field access with defaults
- **`detect_sedumi_version()`** / **`detect_sdpt3_version()`**: Solver-specific version detection

**Benefits of Consolidation**:
- **Single File Deployment**: All MATLAB functionality in one file
- **Reduced Path Dependencies**: No need to manage multiple script paths  
- **Atomic Operations**: All functionality accessible without external dependencies
- **Simplified Maintenance**: Single file to update and version control

---

## Data Flow Pipeline

### Phase 1: Initialization and Validation

```
Python Request → Problem Data Validation → Registry Lookup → Solver Compatibility Check
```

1. **Python Layer**: Validates `ProblemData` structure and required fields
2. **Registry Resolution**: Looks up problem in `problem_registry.yaml` to get file path and type
3. **Compatibility Check**: Verifies solver supports problem type (MAT/DAT-S files)
4. **Environment Check**: Confirms MATLAB availability and solver installation

### Phase 2: MATLAB Execution

```
Temporary File Creation → MATLAB Command Construction → Subprocess Execution
```

1. **Temp File Management**: Creates UUID-based JSON result file
2. **Command Construction**: Builds safe MATLAB command with proper path handling:
   ```matlab
   addpath('matlab_script_dir'); matlab_runner('problem_name', 'solver_name', 'result_file')
   ```
3. **Subprocess Execution**: Runs MATLAB with timeout and captures stdout/stderr

### Phase 3: Problem Resolution and Loading

```
Problem Registry → File Path Resolution → Format-Specific Loading
```

1. **Registry Parsing**: Integrated `read_problem_registry()` extracts problem metadata
2. **Path Resolution**: Converts relative paths to absolute paths
3. **Format Dispatch**: Routes to `mat_loader()` or `dat_loader()` based on file type
4. **Data Extraction**: Returns matrices `A`, `b`, `c` and cone structure `K`

### Phase 4: Solver Execution

```
Input Validation → Solver Configuration → Problem Solving → Status Mapping
```

1. **Input Validation**: Checks matrix dimensions and cone structure consistency
2. **Solver Setup**: Applies minimal configuration for fair benchmarking
3. **Solve Process**: Executes optimization with timing measurement
4. **Result Processing**: Maps solver-specific status codes to standard format

### Phase 5: Metrics Calculation

```
Solution Vectors → Objective Calculation → Feasibility Assessment → Gap Computation
```

1. **Objective Values** (via integrated `calculate_solver_metrics()`): 
   - Primal: `c' * x`
   - Dual: `b' * y`
2. **Infeasibility Measures**:
   - Primal: `||A*x - b|| / (1 + ||b||)`
   - Dual: `sqrt(cone_violation(c - A'*y)) / (1 + ||c||^2)`
3. **Duality Gap**: `|primal_objective - dual_objective|`

### Phase 6: Result Serialization and Return

```
MATLAB Result → JSON Conversion → File Writing → Python Reading → SolverResult Creation
```

1. **JSON Formatting**: Integrated `format_result_to_json()` converts MATLAB struct to JSON string
2. **File Writing**: Integrated `save_json_safely()` writes JSON with error recovery
3. **Python Reading**: `json.load()` parses result file
4. **Result Conversion**: Maps to `SolverResult` object with metadata enhancement

---

## Integration Mechanisms

### Problem Registry Integration

The system uses a centralized YAML configuration to map problem names to file locations:

```yaml
problem_libraries:
  nb:
    display_name: "Network Design"
    file_path: "problems/DIMACS/data/ANTENNA/nb.mat.gz"
    file_type: "mat"
    library_name: "DIMACS"
  
  arch0:
    display_name: "Architecture 0"
    file_path: "problems/SDPLIB/data/arch0.dat-s"
    file_type: "dat-s"
    library_name: "SDPLIB"
```

**Resolution Process**:
1. Python `ProblemData.name` → Registry lookup
2. Extract `file_path` and `file_type`
3. Pass to MATLAB for format-specific loading
4. Load problem data using appropriate loader

### Temporary File Management

**UUID-Based Naming**: Each solver execution creates unique temporary files:
```
/tmp/matlab_sedumi_result_<uuid>.json
```

**Automatic Cleanup**: Context managers ensure cleanup even on exceptions:
```python
with temp_file_context(".json") as result_file:
    # Execute MATLAB solver
    # File automatically cleaned up
```

**Orphan Detection**: Periodic cleanup of abandoned temporary files older than 1 hour.

### Version Detection and Caching

**Multi-Stage Detection**:
1. **Function Existence**: Check if solver functions are available
2. **Version Functions**: Call solver-specific version functions
3. **File Parsing**: Parse version from installation files
4. **Functionality Test**: Verify solver works with minimal problem

**Caching Strategy**: Version information cached during initialization to avoid repeated detection overhead.

---

## Result Processing Chain

### MATLAB Result Structure

```matlab
result = struct();
result.solver_name = 'SeDuMi';
result.solver_version = 'SeDuMi-1.3.7';
result.status = 'optimal';
result.solve_time = 0.245;
result.primal_objective = -4.567;
result.dual_objective = -4.567;
result.gap = 1.234e-10;
result.primal_infeasibility = 2.345e-12;
result.dual_infeasibility = 3.456e-11;
result.iterations = 15;
```

### JSON Conversion Process

**Type Safety**: Integrated `format_result_to_json()` handles MATLAB-specific types:
- `NaN` → `null` (empty array in JSON)
- `±Inf` → `±1e308` 
- Empty arrays → `null`
- Ensure numeric precision preservation

### Python SolverResult Mapping

```python
SolverResult(
    solve_time=matlab_result['solve_time'],
    status=matlab_result['status'].upper(),
    primal_objective_value=safe_float(matlab_result['primal_objective_value']),
    dual_objective_value=safe_float(matlab_result['dual_objective_value']),
    duality_gap=safe_float(matlab_result['duality_gap']),
    primal_infeasibility=safe_float(matlab_result['primal_infeasibility']),
    dual_infeasibility=safe_float(matlab_result['dual_infeasibility']),
    iterations=safe_int(matlab_result['iterations']),
    solver_name='matlab_sedumi',
    solver_version='SeDuMi-1.3.7 (MATLAB R2023b)',
    additional_info={
        'matlab_output': matlab_result,
        'solver_backend': 'sedumi',
        'execution_environment': 'matlab'
    }
)
```

---

## Error Handling Strategy

### Multi-Layer Error Recovery

#### 1. Python Layer
- **Subprocess Failures**: Capture stderr/stdout for diagnostic information
- **Timeout Handling**: Return structured timeout result
- **File I/O Errors**: Handle missing/corrupt result files
- **JSON Parsing**: Graceful handling of malformed JSON

#### 2. MATLAB Layer
- **Solver Errors**: Create error result structure with diagnostic info
- **Problem Loading**: Handle missing files or format errors
- **Metrics Calculation**: Safe handling of NaN/Inf in calculations
- **File Writing**: Atomic operations with rollback on failure

### Error Result Structure

```python
SolverResult.create_error_result(
    error_message="MATLAB execution failed: solver not found",
    solve_time=actual_time_spent,
    solver_name=self.solver_name,
    solver_version=self.get_version()
)
```

### Diagnostic Information

All error results include:
- **Error Classification**: Timeout, solver error, I/O error, etc.
- **Execution Context**: MATLAB version, solver availability, file paths
- **Timing Information**: Time spent before failure
- **Raw Output**: MATLAB stdout/stderr for debugging

---

## Performance Considerations

### Startup Optimization

**MATLAB Initialization**: Cold MATLAB startup can take 5-15 seconds
- **Timeout Adjustment**: Add 15s buffer to solve timeout for startup
- **Startup Warning**: Alert when MATLAB startup > 10s
- **Pre-warming**: Consider pre-warmed MATLAB sessions for production

### Memory Management

**Temporary Files**: 
- UUID-based naming prevents conflicts
- Automatic cleanup prevents disk space issues
- Orphan detection handles interrupted executions

**MATLAB Memory**:
- Release large matrices after use
- Clear temporary variables in MATLAB workspace
- Monitor memory usage in long-running sessions

### Execution Efficiency

**Path Management**: 
- Add MATLAB script paths once per session
- Use absolute paths to avoid working directory issues
- Cache problem registry to avoid repeated YAML parsing

**Subprocess Optimization**:
- Single MATLAB command per solve (no multiple calls)
- Batch multiple problems when possible
- Minimize data serialization overhead

---

## Integration Points Summary

### With Python Benchmark System

1. **SolverInterface Compliance**: Drop-in replacement for other solvers
2. **Problem Registry**: Seamless DIMACS/SDPLIB problem resolution  
3. **Database Integration**: Standard SolverResult format for storage
4. **Report Generation**: Consistent metadata for reporting pipeline

### With External Libraries

1. **DIMACS Problems**: Native MAT file support via `mat_loader.m`
2. **SDPLIB Problems**: DAT-S file support via `dat_loader.m`
3. **Problem Metadata**: Registry-driven problem classification and metadata

### With MATLAB Ecosystem

1. **Solver Detection**: Dynamic discovery of installed solvers
2. **Version Tracking**: Comprehensive version metadata collection
3. **Error Propagation**: Meaningful error messages from MATLAB to Python
4. **Solution Storage**: Optional MAT file output for detailed analysis

This design provides a robust, production-ready integration that maintains the fair benchmarking philosophy while delivering comprehensive MATLAB solver support within the existing Python architecture.