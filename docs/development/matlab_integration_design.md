# MATLAB/Octave Solver Integration - Technical Design

## Overview

This document provides the detailed technical design for integrating MATLAB/Octave optimization solvers (SeDuMi and SDPT3) into the existing optimization solver benchmark system. The design maintains the core principles of fair benchmarking, modular architecture, and production reliability while extending solver coverage to include MATLAB ecosystem.

---

## Design Philosophy

### Core Principles
- **Loose Coupling**: Python-MATLAB integration via command-line execution and JSON data exchange
- **Minimal Configuration**: Use solver defaults to maintain fair benchmarking philosophy
- **Production Reliability**: No disruption to existing 139+ working problems and 9 Python solvers
- **Standardized Interface**: MATLAB solvers implement the same SolverInterface pattern
- **Error Resilience**: Individual MATLAB solver failures don't affect the overall system

### Architecture Strategy
- **Python Entry Point**: Main execution remains in Python for consistency
- **MATLAB Worker Pattern**: MATLAB executes as a worker process with structured input/output
- **JSON Bridge**: Structured data exchange between Python and MATLAB environments
- **Temporary File Management**: Safe handling of intermediate files with automatic cleanup

---

## System Architecture

### High-Level Data Flow
```
Python main.py 
  └── BenchmarkRunner.run_single_benchmark()
      └── MatlabSolver.solve()
          ├── Create temporary JSON input file
          ├── Execute: matlab -batch "matlab_runner('problem', 'solver', 'temp_result.json')"
          ├── Read JSON result file
          ├── Convert to SolverResult
          ├── Cleanup temporary files
          └── Return standardized result
```

### Component Architecture
```
scripts/
├── data_loaders/
│   └── matlab_octave/          # MATLAB data loading functions
│       ├── mat_loader.m        # SeDuMi .mat file loader
│       └── dat_loader.m        # SDPLIB .dat-s file loader
│
├── solvers/
│   └── matlab_octave/          # MATLAB solver implementations
│       ├── matlab_solver.py    # Python interface class
│       ├── sedumi_runner.m     # SeDuMi solver execution
│       ├── sdpt3_runner.m      # SDPT3 solver execution
│       └── matlab_runner.m     # Main MATLAB orchestrator
│
└── benchmark/
    └── runner.py              # Extended to support MATLAB solvers
```

---

## Detailed Component Design

### 1. MATLAB Data Loaders

#### `scripts/data_loaders/matlab_octave/mat_loader.m`
```matlab
function [A, b, c, K] = mat_loader(file_path)
% Load SeDuMi format .mat file and extract optimization problem data
%
% Input:
%   file_path: Path to .mat file containing SeDuMi format data
%
% Output:
%   A: Constraint matrix (sparse)
%   b: Right-hand side vector
%   c: Objective vector
%   K: Cone structure
%
% The function handles:
% - Compressed .mat.gz files (automatic decompression)
% - Standard .mat files
% - Error handling for corrupted files

try
    % Load .mat file (handles .gz automatically in modern MATLAB)
    data = load(file_path);
    
    % Extract SeDuMi format fields
    if isfield(data, 'A') && isfield(data, 'b') && isfield(data, 'c')
        A = data.A;
        b = data.b;
        c = data.c;
        
        % Extract or construct cone structure
        if isfield(data, 'K')
            K = data.K;
        else
            % Default cone structure for problems without explicit cones
            K = struct();
            K.f = 0;  % Number of free variables
            K.l = length(c);  % Number of linear inequality constraints
        end
    else
        error('Invalid .mat file: missing required fields A, b, or c');
    end
    
catch ME
    error('Failed to load .mat file: %s', ME.message);
end
end
```

#### `scripts/data_loaders/matlab_octave/dat_loader.m`
```matlab
function [A, b, c, K] = dat_loader(file_path)
% Load SDPLIB format .dat-s file and convert to SeDuMi format
%
% Input:
%   file_path: Path to .dat-s file containing SDPA sparse format data
%
% Output:
%   A: Constraint matrix (sparse)
%   b: Right-hand side vector  
%   c: Objective vector
%   K: Cone structure
%
% The function parses SDPA sparse format and converts to SeDuMi format

try
    % Open file for reading
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Cannot open file: %s', file_path);
    end
    
    % Parse SDPA header
    m = fscanf(fid, '%d', 1);  % Number of constraints
    nblocks = fscanf(fid, '%d', 1);  % Number of blocks
    
    % Read block sizes
    block_sizes = fscanf(fid, '%d', nblocks);
    
    % Read objective vector c
    c = fscanf(fid, '%f', m);
    
    % Initialize matrices
    total_vars = sum(block_sizes.^2);  % Total variables for SDP blocks
    A = sparse(m, total_vars);
    
    % Parse constraint matrices
    while ~feof(fid)
        line = fgets(fid);
        if ischar(line)
            data = sscanf(line, '%d %d %d %d %f');
            if length(data) == 5
                % Process matrix entry: constraint, block, row, col, value
                % Convert to SeDuMi format indexing
                % ... (detailed parsing implementation)
            end
        end
    end
    
    fclose(fid);
    
    % Construct cone structure for SDP
    K = struct();
    K.f = 0;
    K.l = 0;
    K.s = block_sizes;  % SDP block sizes
    
    % Create right-hand side vector (typically zeros for feasibility)
    b = zeros(m, 1);
    
catch ME
    if exist('fid', 'var') && fid ~= -1
        fclose(fid);
    end
    error('Failed to load .dat-s file: %s', ME.message);
end
end
```

### 2. MATLAB Solver Runners

#### `scripts/solvers/matlab_octave/sedumi_runner.m`
```matlab
function result = sedumi_runner(A, b, c, K, solver_options)
% Execute SeDuMi solver and collect standardized metrics
%
% Input:
%   A, b, c, K: SeDuMi format optimization problem
%   solver_options: Optional solver parameters (struct)
%
% Output:
%   result: Struct with standardized solver metrics

% Set default options for fair benchmarking
if nargin < 5 || isempty(solver_options)
    solver_options = struct();
end

% Apply minimal configuration (fair benchmarking principle)
pars = struct();
pars.fid = 0;  % Suppress output for clean benchmarking
pars.eps = 1e-8;  % Default tolerance
pars.bigeps = 1e-3;  % Default feasibility tolerance

% Override with user options if provided
if isfield(solver_options, 'eps')
    pars.eps = solver_options.eps;
end

try
    % Record start time
    start_time = tic;
    
    % Execute SeDuMi solver
    [x, y, info] = sedumi(A, b, c, K, pars);
    
    % Record solve time
    solve_time = toc(start_time);
    
    % Extract solver information
    solver_version = sedumi_version();
    matlab_version = version();
    
    % Compute standardized metrics
    result = struct();
    result.solve_time = solve_time;
    
    % Map SeDuMi status to standard format
    if info.pinf == 0 && info.dinf == 0
        result.status = 'optimal';
        result.primal_objective_value = c' * x;
        result.dual_objective_value = b' * y;
        result.duality_gap = abs(result.primal_objective_value - result.dual_objective_value);
    elseif info.pinf == 1
        result.status = 'primal_infeasible';
        result.primal_objective_value = [];
        result.dual_objective_value = [];
        result.duality_gap = [];
    elseif info.dinf == 1
        result.status = 'dual_infeasible';
        result.primal_objective_value = [];
        result.dual_objective_value = [];
        result.duality_gap = [];
    else
        result.status = 'unknown';
        result.primal_objective_value = [];
        result.dual_objective_value = [];
        result.duality_gap = [];
    end
    
    % Extract infeasibility measures
    result.primal_infeasibility = info.numerr;
    result.dual_infeasibility = info.numerr;
    result.iterations = info.iter;
    
    % Store solver metadata
    result.solver_version = solver_version;
    result.matlab_version = matlab_version;
    result.additional_info = info;
    
catch ME
    % Handle solver errors gracefully
    result = struct();
    result.solve_time = 0;
    result.status = 'error';
    result.primal_objective_value = [];
    result.dual_objective_value = [];
    result.duality_gap = [];
    result.primal_infeasibility = [];
    result.dual_infeasibility = [];
    result.iterations = [];
    result.solver_version = 'unknown';
    result.matlab_version = version();
    result.error_message = ME.message;
end
end
```

#### `scripts/solvers/matlab_octave/sdpt3_runner.m`
```matlab
function result = sdpt3_runner(A, b, c, K, solver_options)
% Execute SDPT3 solver and collect standardized metrics
%
% Input:
%   A, b, c, K: SeDuMi format optimization problem
%   solver_options: Optional solver parameters (struct)
%
% Output:
%   result: Struct with standardized solver metrics

% Set default options for fair benchmarking
if nargin < 5 || isempty(solver_options)
    solver_options = struct();
end

% SDPT3 default options (minimal configuration)
options = sqlparameters;
options.printlevel = 0;  % Suppress output
options.gaptol = 1e-8;   % Default duality gap tolerance
options.inftol = 1e-8;   % Default infeasibility tolerance

% Override with user options if provided
if isfield(solver_options, 'gaptol')
    options.gaptol = solver_options.gaptol;
end

try
    % Record start time
    start_time = tic;
    
    % Execute SDPT3 solver
    [blk, A_sdpt3, C, b_sdpt3] = read_sedumi(A, b, c, K);
    [obj, X, y, Z, info, runhist] = sqlp(blk, A_sdpt3, C, b_sdpt3, options);
    
    % Record solve time
    solve_time = toc(start_time);
    
    % Get solver version information
    matlab_version = version();
    sdpt3_version = '4.0';  % Default version (can be detected if available)
    
    % Compute standardized metrics
    result = struct();
    result.solve_time = solve_time;
    
    % Map SDPT3 status to standard format
    if info.termcode == 0
        result.status = 'optimal';
        result.primal_objective_value = obj(1);
        result.dual_objective_value = obj(2);
        result.duality_gap = abs(obj(1) - obj(2));
    elseif info.termcode == 1
        result.status = 'primal_infeasible';
        result.primal_objective_value = [];
        result.dual_objective_value = [];
        result.duality_gap = [];
    elseif info.termcode == 2
        result.status = 'dual_infeasible';
        result.primal_objective_value = [];
        result.dual_objective_value = [];
        result.duality_gap = [];
    else
        result.status = 'unknown';
        result.primal_objective_value = [];
        result.dual_objective_value = [];
        result.duality_gap = [];
    end
    
    % Extract infeasibility measures
    result.primal_infeasibility = info.pinfeas;
    result.dual_infeasibility = info.dinfeas;
    result.iterations = info.iter;
    
    % Store solver metadata
    result.solver_version = sdpt3_version;
    result.matlab_version = matlab_version;
    result.additional_info = info;
    
catch ME
    % Handle solver errors gracefully
    result = struct();
    result.solve_time = 0;
    result.status = 'error';
    result.primal_objective_value = [];
    result.dual_objective_value = [];
    result.duality_gap = [];
    result.primal_infeasibility = [];
    result.dual_infeasibility = [];
    result.iterations = [];
    result.solver_version = 'unknown';
    result.matlab_version = version();
    result.error_message = ME.message;
end
end
```

### 3. MATLAB Integration Orchestrator

#### `scripts/solvers/matlab_octave/matlab_runner.m`
```matlab
function matlab_runner(problem_name, solver_name, result_file)
% Main MATLAB orchestrator for benchmark execution
%
% Input:
%   problem_name: Name of problem from problem_registry.yaml
%   solver_name: Name of solver ('sedumi' or 'sdpt3')
%   result_file: Path to output JSON file for results
%
% This function:
% 1. Loads problem_registry.yaml configuration
% 2. Resolves problem file path and type
% 3. Loads problem data using appropriate loader
% 4. Executes specified solver
% 5. Saves results to JSON file

try
    % Add necessary paths for solvers and loaders
    addpath(genpath('.'));
    
    % Load problem registry configuration
    config = load_problem_registry();
    
    % Validate problem exists
    if ~isfield(config.problem_libraries, problem_name)
        error('Unknown problem: %s', problem_name);
    end
    
    problem_config = config.problem_libraries.(problem_name);
    
    % Resolve file path
    file_path = problem_config.file_path;
    file_type = problem_config.file_type;
    
    % Load problem data using appropriate loader
    if strcmp(file_type, 'mat')
        [A, b, c, K] = mat_loader(file_path);
    elseif strcmp(file_type, 'dat-s')
        [A, b, c, K] = dat_loader(file_path);
    else
        error('Unsupported file type: %s', file_type);
    end
    
    % Execute solver
    if strcmp(solver_name, 'sedumi')
        result = sedumi_runner(A, b, c, K);
    elseif strcmp(solver_name, 'sdpt3')
        result = sdpt3_runner(A, b, c, K);
    else
        error('Unknown solver: %s', solver_name);
    end
    
    % Convert result to JSON-compatible format
    json_result = struct();
    json_result.solve_time = result.solve_time;
    json_result.status = result.status;
    
    % Handle optional numeric fields (convert [] to null)
    if isempty(result.primal_objective_value)
        json_result.primal_objective_value = [];
    else
        json_result.primal_objective_value = result.primal_objective_value;
    end
    
    if isempty(result.dual_objective_value)
        json_result.dual_objective_value = [];
    else
        json_result.dual_objective_value = result.dual_objective_value;
    end
    
    if isempty(result.duality_gap)
        json_result.duality_gap = [];
    else
        json_result.duality_gap = result.duality_gap;
    end
    
    if isempty(result.primal_infeasibility)
        json_result.primal_infeasibility = [];
    else
        json_result.primal_infeasibility = result.primal_infeasibility;
    end
    
    if isempty(result.dual_infeasibility)
        json_result.dual_infeasibility = [];
    else
        json_result.dual_infeasibility = result.dual_infeasibility;
    end
    
    if isempty(result.iterations)
        json_result.iterations = [];
    else
        json_result.iterations = result.iterations;
    end
    
    json_result.solver_version = result.solver_version;
    json_result.matlab_version = result.matlab_version;
    
    % Save result to JSON file
    json_text = jsonencode(json_result);
    fid = fopen(result_file, 'w');
    if fid == -1
        error('Cannot create result file: %s', result_file);
    end
    fprintf(fid, '%s', json_text);
    fclose(fid);
    
    fprintf('MATLAB solver execution completed successfully\n');
    
catch ME
    % Save error result to JSON file
    error_result = struct();
    error_result.solve_time = 0;
    error_result.status = 'error';
    error_result.primal_objective_value = [];
    error_result.dual_objective_value = [];
    error_result.duality_gap = [];
    error_result.primal_infeasibility = [];
    error_result.dual_infeasibility = [];
    error_result.iterations = [];
    error_result.solver_version = 'unknown';
    error_result.matlab_version = version();
    error_result.error_message = ME.message;
    
    json_text = jsonencode(error_result);
    fid = fopen(result_file, 'w');
    if fid ~= -1
        fprintf(fid, '%s', json_text);
        fclose(fid);
    end
    
    fprintf('MATLAB solver execution failed: %s\n', ME.message);
    exit(1);  % Exit with error code
end

% Exit successfully
exit(0);
end

function config = load_problem_registry()
% Load problem registry YAML configuration
% This is a simplified YAML parser for the specific structure we need

config_file = 'config/problem_registry.yaml';
if ~exist(config_file, 'file')
    error('Problem registry file not found: %s', config_file);
end

% Read YAML file (basic parsing for our structure)
% In practice, this would use a proper YAML parser
% For now, assume we can access the data structure directly

% Placeholder: In real implementation, parse YAML or use JSON equivalent
config = struct();
config.problem_libraries = struct();

% This would be replaced with actual YAML parsing
% config = yaml.loadFile(config_file);
end
```

### 4. Python Integration Interface

#### `scripts/solvers/matlab_octave/matlab_solver.py`
```python
"""
MATLAB Solver Integration for Optimization Benchmark System.

This module provides a Python interface for MATLAB optimization solvers (SeDuMi, SDPT3)
that integrates with the existing benchmark system architecture. It uses command-line
execution and JSON data exchange for loose coupling between Python and MATLAB.
"""

import os
import sys
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverInterface, SolverResult
from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("matlab_solver")


class MatlabSolver(SolverInterface):
    """Python interface for MATLAB optimization solvers."""
    
    SUPPORTED_SOLVERS = {
        'sedumi': 'SeDuMi',
        'sdpt3': 'SDPT3'
    }
    
    def __init__(self, matlab_solver: str, matlab_executable: str = 'matlab',
                 timeout: Optional[float] = 300, use_octave: bool = False, **kwargs):
        """
        Initialize MATLAB solver interface.
        
        Args:
            matlab_solver: MATLAB solver name ('sedumi' or 'sdpt3')
            matlab_executable: Path to MATLAB executable
            timeout: Solver timeout in seconds
            use_octave: Use Octave instead of MATLAB
            **kwargs: Additional configuration parameters
        """
        if matlab_solver not in self.SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported MATLAB solver: {matlab_solver}. "
                           f"Supported: {list(self.SUPPORTED_SOLVERS.keys())}")
        
        # Generate solver name for registration
        solver_name = f"matlab_{matlab_solver}"
        
        super().__init__(solver_name, matlab_solver=matlab_solver, 
                        matlab_executable=matlab_executable, timeout=timeout, **kwargs)
        
        self.matlab_solver = matlab_solver
        self.matlab_executable = matlab_executable
        self.timeout = timeout
        self.use_octave = use_octave
        
        # Verify MATLAB/Octave availability
        self._verify_matlab_availability()
        
        logger.info(f"Initialized MATLAB solver '{self.solver_name}' "
                   f"using {matlab_solver} via {matlab_executable}")
    
    def _verify_matlab_availability(self) -> None:
        """Verify that MATLAB/Octave is available and can execute."""
        try:
            cmd = [self.matlab_executable, '-batch', 'disp("MATLAB OK")']
            if self.use_octave:
                cmd = [self.matlab_executable, '--eval', 'disp("Octave OK")']
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"MATLAB/Octave execution failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MATLAB/Octave verification timed out")
        except FileNotFoundError:
            raise RuntimeError(f"MATLAB/Octave executable not found: {self.matlab_executable}")
    
    def solve(self, problem_data: ProblemData, timeout: Optional[float] = None) -> SolverResult:
        """
        Solve optimization problem using MATLAB solver.
        
        Args:
            problem_data: Problem data in unified format
            timeout: Optional timeout override
            
        Returns:
            SolverResult with standardized fields
        """
        solve_timeout = timeout or self.timeout
        start_time = time.time()
        
        # Generate unique temporary file names
        temp_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        result_file = os.path.join(temp_dir, f"matlab_result_{temp_id}.json")
        
        try:
            # Verify problem data has required SeDuMi format fields
            if not all(hasattr(problem_data, field) for field in ['A', 'b', 'c', 'K']):
                return SolverResult.create_error_result(
                    "Problem data missing required SeDuMi format fields (A, b, c, K)",
                    solve_time=time.time() - start_time,
                    solver_name=self.solver_name,
                    solver_version=self.get_version()
                )
            
            # Get problem name from metadata or use default
            problem_name = getattr(problem_data, 'name', 'unknown_problem')
            
            # Execute MATLAB solver via command line
            matlab_command = f"matlab_runner('{problem_name}', '{self.matlab_solver}', '{result_file}')"
            
            if self.use_octave:
                cmd = [self.matlab_executable, '--eval', matlab_command]
            else:
                cmd = [self.matlab_executable, '-batch', matlab_command]
            
            logger.debug(f"Executing MATLAB command: {' '.join(cmd)}")
            
            # Execute with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=solve_timeout,
                cwd=project_root
            )
            
            solve_time = time.time() - start_time
            
            # Check execution success
            if result.returncode != 0:
                error_msg = f"MATLAB execution failed (code {result.returncode}): {result.stderr}"
                logger.error(error_msg)
                return SolverResult.create_error_result(
                    error_msg,
                    solve_time=solve_time,
                    solver_name=self.solver_name,
                    solver_version=self.get_version()
                )
            
            # Read JSON result file
            if not os.path.exists(result_file):
                return SolverResult.create_error_result(
                    "MATLAB solver did not produce result file",
                    solve_time=solve_time,
                    solver_name=self.solver_name,
                    solver_version=self.get_version()
                )
            
            with open(result_file, 'r') as f:
                matlab_result = json.load(f)
            
            # Convert MATLAB result to SolverResult
            return self._convert_matlab_result(matlab_result, solve_time)
            
        except subprocess.TimeoutExpired:
            return SolverResult.create_timeout_result(
                solve_timeout,
                solver_name=self.solver_name,
                solver_version=self.get_version()
            )
            
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"MATLAB solver execution failed: {e}")
            return SolverResult.create_error_result(
                str(e),
                solve_time=solve_time,
                solver_name=self.solver_name,
                solver_version=self.get_version()
            )
            
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(result_file):
                    os.remove(result_file)
            except:
                pass  # Ignore cleanup failures
    
    def _convert_matlab_result(self, matlab_result: Dict[str, Any], solve_time: float) -> SolverResult:
        """Convert MATLAB JSON result to SolverResult format."""
        
        # Extract solver version information
        solver_version = matlab_result.get('solver_version', 'unknown')
        matlab_version = matlab_result.get('matlab_version', 'unknown')
        combined_version = f"{solver_version} (MATLAB {matlab_version})"
        
        # Handle None/null values from JSON
        def safe_float(value):
            return None if value is None or value == [] else float(value)
        
        def safe_int(value):
            return None if value is None or value == [] else int(value)
        
        try:
            return SolverResult(
                solve_time=solve_time,
                status=matlab_result.get('status', 'unknown').upper(),
                primal_objective_value=safe_float(matlab_result.get('primal_objective_value')),
                dual_objective_value=safe_float(matlab_result.get('dual_objective_value')),
                duality_gap=safe_float(matlab_result.get('duality_gap')),
                primal_infeasibility=safe_float(matlab_result.get('primal_infeasibility')),
                dual_infeasibility=safe_float(matlab_result.get('dual_infeasibility')),
                iterations=safe_int(matlab_result.get('iterations')),
                solver_name=self.solver_name,
                solver_version=combined_version,
                additional_info={
                    'matlab_output': matlab_result,
                    'matlab_version': matlab_version
                }
            )
        except Exception as e:
            # If conversion fails, return error result
            return SolverResult.create_error_result(
                f"Failed to convert MATLAB result: {e}",
                solve_time=solve_time,
                solver_name=self.solver_name,
                solver_version=combined_version
            )
    
    def get_version(self) -> str:
        """Get MATLAB solver version information."""
        try:
            # Try to get version from MATLAB
            cmd_map = {
                'sedumi': "disp(sedumi_version())",
                'sdpt3': "disp('SDPT3-4.0')"  # Default version
            }
            
            cmd_str = cmd_map.get(self.matlab_solver, "disp('unknown')")
            
            if self.use_octave:
                cmd = [self.matlab_executable, '--eval', cmd_str]
            else:
                cmd = [self.matlab_executable, '-batch', cmd_str]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=project_root
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                return f"{self.SUPPORTED_SOLVERS[self.matlab_solver]} {version}"
            else:
                return f"{self.SUPPORTED_SOLVERS[self.matlab_solver]} (version unknown)"
                
        except Exception:
            return f"{self.SUPPORTED_SOLVERS[self.matlab_solver]} (version detection failed)"
    
    def validate_problem_compatibility(self, problem_data: ProblemData) -> bool:
        """Check if problem is compatible with MATLAB solver."""
        # Check for required SeDuMi format fields
        required_fields = ['A', 'b', 'c', 'K']
        has_required = all(hasattr(problem_data, field) for field in required_fields)
        
        if not has_required:
            logger.warning(f"Problem missing required SeDuMi fields for {self.solver_name}")
            return False
        
        # Check problem type compatibility
        problem_type = getattr(problem_data, 'problem_type', '').upper()
        
        # Both SeDuMi and SDPT3 support LP, QP, SOCP, SDP
        supported_types = ['LP', 'QP', 'SOCP', 'SDP']
        
        if problem_type not in supported_types:
            logger.warning(f"Problem type {problem_type} not supported by {self.solver_name}")
            return False
        
        return True


class SeDuMiSolver(MatlabSolver):
    """Convenience class for SeDuMi solver."""
    
    def __init__(self, **kwargs):
        super().__init__(matlab_solver='sedumi', **kwargs)


class SDPT3Solver(MatlabSolver):
    """Convenience class for SDPT3 solver."""
    
    def __init__(self, **kwargs):
        super().__init__(matlab_solver='sdpt3', **kwargs)
```

---

## Configuration Integration

### 1. Solver Registry Extension

Update `config/solver_registry.yaml`:
```yaml
# Existing Python solvers
solvers:
  scipy_linprog:
    display_name: "SciPy linprog"
    
  cvxpy_clarabel:
    display_name: "CLARABEL (via CVXPY)"
    
  # ... other existing solvers ...
  
  # New MATLAB solvers
  matlab_sedumi:
    display_name: "SeDuMi (MATLAB)"
    
  matlab_sdpt3:
    display_name: "SDPT3 (MATLAB)"
```

### 2. Benchmark Runner Integration

Extend `scripts/benchmark/runner.py`:
```python
# Add import
from scripts.solvers.matlab_octave.matlab_solver import SeDuMiSolver, SDPT3Solver

# Update create_solver method
def create_solver(self, solver_name: str) -> SolverInterface:
    """Create solver instance based on solver name"""
    
    # Existing solver creation logic...
    
    # Add MATLAB solver support
    elif solver_name == "matlab_sedumi":
        return SeDuMiSolver()
    elif solver_name == "matlab_sdpt3":
        return SDPT3Solver()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
```

---

## Testing Strategy

### 1. Unit Testing

#### MATLAB Function Tests
- **Data Loader Tests**: Verify mat_loader.m and dat_loader.m correctly parse files
- **Solver Runner Tests**: Test sedumi_runner.m and sdpt3_runner.m with simple problems
- **Integration Tests**: Verify matlab_runner.m orchestrates correctly

#### Python Integration Tests
- **Command Execution**: Test MATLAB command line execution and JSON parsing
- **Error Handling**: Verify graceful handling of MATLAB failures
- **Timeout Handling**: Test solver timeout and cleanup

### 2. Integration Testing

#### Small Problem Tests
- **Test Problems**: Use nb (DIMACS) and arch0 (SDPLIB) for initial validation
- **Result Consistency**: Compare MATLAB solver results with CVXPY solvers on same problems
- **Performance Baseline**: Establish execution time baselines

#### End-to-End Workflow
- **Full Pipeline**: Test complete Python → MATLAB → Database workflow
- **Error Scenarios**: Test handling of corrupted files, solver failures, timeouts
- **Cleanup Verification**: Ensure temporary files are properly removed

### 3. Performance Testing

#### Execution Time Analysis
- **Overhead Measurement**: Quantify Python-MATLAB communication overhead
- **Solver Performance**: Compare native MATLAB performance vs CVXPY equivalents
- **Scalability Testing**: Test with larger problems from DIMACS/SDPLIB

---

## Error Handling Strategy

### 1. MATLAB Environment Errors
- **Installation Detection**: Verify MATLAB/Octave availability at startup
- **Path Configuration**: Validate solver paths and dependencies
- **License Issues**: Handle MATLAB license failures gracefully

### 2. Execution Errors
- **Solver Failures**: Capture MATLAB solver errors and convert to standard format
- **Memory Issues**: Handle large problems that exceed MATLAB memory limits
- **Timeout Handling**: Clean termination of long-running MATLAB processes

### 3. Data Exchange Errors
- **File I/O Errors**: Handle temporary file creation/deletion failures
- **JSON Parsing**: Validate JSON format and handle parsing errors
- **Type Conversion**: Safe conversion between MATLAB and Python data types

### 4. Graceful Degradation
- **Individual Failures**: System continues if one MATLAB solver fails
- **Fallback Behavior**: Option to skip MATLAB solvers if environment unavailable
- **Error Reporting**: Clear error messages for debugging and user guidance

---

## Performance Considerations

### 1. Startup Overhead
- **MATLAB Initialization**: MATLAB startup can be slow (~2-5 seconds)
- **Mitigation**: Consider MATLAB session persistence for multiple problems
- **Alternative**: Use Octave for faster startup times

### 2. Memory Management
- **Large Problems**: MATLAB may require more memory than Python solvers
- **Cleanup**: Ensure proper cleanup of MATLAB variables and temporary files
- **Monitoring**: Track memory usage during MATLAB execution

### 3. Parallel Execution
- **Process Isolation**: Each MATLAB solver runs in separate process
- **Resource Limits**: Consider MATLAB license limits for parallel execution
- **Coordination**: Manage temporary file conflicts in parallel scenarios

---

## Security Considerations

### 1. Command Injection Prevention
- **Input Sanitization**: Validate all inputs passed to MATLAB command line
- **Path Validation**: Ensure file paths are within expected directories
- **Command Construction**: Use safe command construction methods

### 2. Temporary File Security
- **File Permissions**: Set appropriate permissions on temporary files
- **Cleanup Assurance**: Ensure cleanup even in error scenarios
- **Name Collision**: Use unique temporary file names

### 3. Environment Isolation
- **Working Directory**: Execute MATLAB in controlled working directory
- **Path Restrictions**: Limit MATLAB path access to required directories
- **Resource Limits**: Apply timeout and memory limits to MATLAB processes

---

## Deployment Considerations

### 1. Environment Setup
- **MATLAB Installation**: Require MATLAB R2020a+ or Octave 6.0+
- **Solver Installation**: Document SeDuMi and SDPT3 installation procedures
- **Path Configuration**: Ensure solver paths are correctly configured

### 2. CI/CD Integration
- **GitHub Actions**: Consider MATLAB availability in CI environment
- **Octave Alternative**: Use Octave for CI if MATLAB unavailable
- **Skip Strategy**: Graceful handling when MATLAB solvers unavailable

### 3. Documentation
- **Installation Guide**: Step-by-step MATLAB/Octave setup instructions
- **Troubleshooting**: Common issues and solutions
- **Performance Tuning**: Optimization recommendations

---

## Future Enhancements

### 1. Additional Solvers
- **MOSEK**: Add MOSEK MATLAB interface
- **CVX**: Integrate CVX modeling language
- **YALMIP**: Support YALMIP optimization framework

### 2. Performance Optimization
- **Session Persistence**: Maintain MATLAB session for multiple problems
- **Batch Processing**: Process multiple problems in single MATLAB session
- **Memory Optimization**: Optimize memory usage for large problems

### 3. Enhanced Integration
- **Native MEX**: Consider MEX interface for tighter integration
- **Direct Data Transfer**: Binary data transfer instead of JSON
- **Streaming Results**: Real-time result streaming for long-running problems

---

*This technical design provides a comprehensive foundation for integrating MATLAB/Octave solvers while maintaining the system's core principles of reliability, modularity, and fair benchmarking.*

*Last Updated: December 2025*