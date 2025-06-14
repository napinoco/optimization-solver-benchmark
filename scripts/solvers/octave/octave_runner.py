"""
Octave Solver Runner
===================

Executes optimization problems using GNU Octave with MATLAB-compatible syntax.
Provides interface to Octave's optimization toolbox functions.
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.solver_interface import SolverInterface, SolverResult
from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("octave_solver")


class OctaveSolver(SolverInterface):
    """GNU Octave-based solver for LP and QP problems."""
    
    def __init__(self, name: str = "Octave", timeout: float = 300.0, 
                 octave_path: Optional[str] = None, options: Optional[Dict] = None):
        """
        Initialize Octave solver.
        
        Args:
            name: Solver name
            timeout: Timeout in seconds
            octave_path: Path to octave executable
            options: Additional options for the solver
        """
        super().__init__(name, timeout)
        self.octave_path = octave_path or self._find_octave()
        self.options = options or {}
        self.octave_dir = Path(__file__).parent
        self.logger.info(f"Initialized Octave solver with path '{self.octave_path}'")
        
        # Verify Octave installation
        if not self._verify_octave():
            raise RuntimeError("Octave not found or not properly installed")
    
    def _find_octave(self) -> str:
        """Find Octave executable in PATH."""
        
        # Common Octave executable names
        octave_names = ['octave', 'octave-cli', 'octave.exe']
        
        for name in octave_names:
            try:
                result = subprocess.run(['which', name], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        # Try common installation paths
        common_paths = [
            '/usr/bin/octave',
            '/usr/local/bin/octave',
            '/opt/homebrew/bin/octave',
            'C:\\Program Files\\GNU Octave\\Octave-*\\mingw64\\bin\\octave.exe'
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return 'octave'  # Default fallback
    
    def _verify_octave(self) -> bool:
        """Verify Octave installation and basic functionality."""
        
        try:
            # Test basic Octave execution
            result = subprocess.run(
                [self.octave_path, '--eval', 'printf("Octave OK\\n");'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and "Octave OK" in result.stdout:
                self.logger.debug("Octave installation verified")
                return True
            else:
                self.logger.error(f"Octave verification failed: {result.stderr}")
                return False
                
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            self.logger.error(f"Could not execute Octave: {e}")
            return False
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """
        Solve optimization problem using Octave.
        
        Args:
            problem: Problem data to solve
            
        Returns:
            SolverResult containing solve status and results
        """
        self.logger.debug(f"Solving {problem.problem_class} problem '{problem.name}' with Octave")
        
        start_time = time.time()
        
        try:
            # Create temporary files for problem data and results
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write problem data to temporary files
                problem_file = temp_path / "problem_data.json"
                result_file = temp_path / "result_data.json"
                
                self._write_problem_data(problem, problem_file)
                
                # Choose appropriate Octave solver based on problem type
                if problem.problem_class == "LP":
                    octave_script = self._get_linprog_script()
                elif problem.problem_class == "QP":
                    octave_script = self._get_quadprog_script()
                else:
                    raise ValueError(f"Unsupported problem type: {problem.problem_class}")
                
                # Execute Octave solver
                success, output, error = self._execute_octave_script(
                    octave_script, problem_file, result_file
                )
                
                solve_time = time.time() - start_time
                
                if success and result_file.exists():
                    # Read results from Octave
                    result_data = self._read_result_data(result_file)
                    
                    return SolverResult(
                        solver_name=self.name,
                        problem_name=problem.name,
                        status=result_data.get('status', 'unknown'),
                        objective_value=result_data.get('objective_value'),
                        solve_time=solve_time,
                        iterations=result_data.get('iterations'),
                        duality_gap=result_data.get('duality_gap')
                    )
                else:
                    # Handle execution failure
                    self.logger.error(f"Octave execution failed: {error}")
                    return SolverResult(
                        solver_name=self.name,
                        problem_name=problem.name,
                        status='error',
                        solve_time=solve_time,
                        error_message=f"Octave execution failed: {error}"
                    )
        
        except Exception as e:
            solve_time = time.time() - start_time
            self.logger.error(f"Error solving problem with Octave: {e}")
            return SolverResult(
                solver_name=self.name,
                problem_name=problem.name,
                status='error',
                solve_time=solve_time,
                error_message=str(e)
            )
    
    def _write_problem_data(self, problem: ProblemData, filename: Path) -> None:
        """Write problem data to JSON file for Octave to read."""
        
        problem_data = {
            'name': problem.name,
            'problem_class': problem.problem_class,
            'A_eq': problem.A_eq.tolist() if problem.A_eq is not None else None,
            'b_eq': problem.b_eq.tolist() if problem.b_eq is not None else None,
            'A_ineq': problem.A_ub.tolist() if problem.A_ub is not None else None,  # Use A_ub
            'b_ineq': problem.b_ub.tolist() if problem.b_ub is not None else None,  # Use b_ub
            'c': problem.c.tolist() if problem.c is not None else None,
            'Q': problem.P.tolist() if problem.P is not None else None,  # Use P for quadratic matrix
            'bounds': problem.bounds,
            'n_variables': problem.metadata.get('n_variables') if problem.metadata else None,
            'n_constraints': problem.metadata.get('n_constraints') if problem.metadata else None
        }
        
        with open(filename, 'w') as f:
            json.dump(problem_data, f, indent=2)
    
    def _read_result_data(self, filename: Path) -> Dict[str, Any]:
        """Read results from JSON file written by Octave."""
        
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read Octave results: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _execute_octave_script(self, script_content: str, problem_file: Path, 
                              result_file: Path) -> tuple[bool, str, str]:
        """Execute Octave script with problem data."""
        
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as script_file:
                # Write script with file paths
                full_script = script_content.format(
                    problem_file=str(problem_file).replace('\\', '/'),
                    result_file=str(result_file).replace('\\', '/'),
                    octave_dir=str(self.octave_dir).replace('\\', '/')
                )
                script_file.write(full_script)
                script_file.flush()
                
                # Execute Octave script
                cmd = [
                    self.octave_path,
                    '--eval', f"run('{script_file.name}');"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                # Clean up script file
                os.unlink(script_file.name)
                
                return (
                    result.returncode == 0,
                    result.stdout,
                    result.stderr
                )
        
        except subprocess.TimeoutExpired:
            return False, "", "Timeout expired"
        except Exception as e:
            return False, "", str(e)
    
    def _get_linprog_script(self) -> str:
        """Get Octave script for linear programming."""
        
        return '''
% Linear Programming Solver for Octave
% Load problem data
addpath('{octave_dir}');
problem_data = loadjson('{problem_file}');

% Extract problem matrices
if ~isempty(problem_data.c)
    c = problem_data.c(:);
else
    error('Linear programming requires objective vector c');
end

% Combine inequality and equality constraints for linprog
A = [];
b = [];
Aeq = [];
beq = [];

if ~isempty(problem_data.A_ineq) && ~isempty(problem_data.b_ineq)
    A = problem_data.A_ineq;
    b = problem_data.b_ineq(:);
end

if ~isempty(problem_data.A_eq) && ~isempty(problem_data.b_eq)
    Aeq = problem_data.A_eq;
    beq = problem_data.b_eq(:);
end

% Set up bounds
lb = [];
ub = [];
if ~isempty(problem_data.bounds)
    n_vars = length(c);
    lb = -Inf(n_vars, 1);
    ub = Inf(n_vars, 1);
    
    for i = 1:length(problem_data.bounds)
        if ~isempty(problem_data.bounds{{i}})
            if length(problem_data.bounds{{i}}) >= 1 && ~isempty(problem_data.bounds{{i}}{{1}})
                lb(i) = problem_data.bounds{{i}}{{1}};
            end
            if length(problem_data.bounds{{i}}) >= 2 && ~isempty(problem_data.bounds{{i}}{{2}})
                ub(i) = problem_data.bounds{{i}}{{2}};
            end
        end
    end
end

% Solve using linprog (if available) or alternative solver
try
    if exist('linprog', 'file')
        % Use Octave's linprog if available
        [x, fval, exitflag, output] = linprog(c, A, b, Aeq, beq, lb, ub);
        
        % Map exit flags to status
        if exitflag == 1
            status = 'optimal';
        elseif exitflag == 0
            status = 'max_iter';
        elseif exitflag == -2
            status = 'infeasible';
        elseif exitflag == -3
            status = 'unbounded';
        else
            status = 'error';
        end
        
        iterations = [];
        if isfield(output, 'iterations')
            iterations = output.iterations;
        end
        
    else
        % Fallback to basic implementation
        warning('linprog not available, using basic solver');
        
        % Simple equality-constrained LP solver
        if isempty(A) && ~isempty(Aeq)
            % Equality constrained: min c'x s.t. Aeq*x = beq
            if rank(Aeq) == size(Aeq, 1)
                % Full rank, use least squares solution
                x = pinv(Aeq) * beq;
                fval = c' * x;
                status = 'optimal';
                iterations = 1;
            else
                status = 'infeasible';
                x = [];
                fval = NaN;
                iterations = 0;
            end
        else
            % More complex case - use simple method
            status = 'error';
            x = [];
            fval = NaN;
            iterations = 0;
        end
    end
    
    % Prepare result data
    result.status = status;
    result.objective_value = fval;
    result.iterations = iterations;
    result.solution = x;
    
catch err
    result.status = 'error';
    result.error = err.message;
    result.objective_value = NaN;
    result.iterations = 0;
end

% Save results
savejson(result, '{result_file}');
'''
    
    def _get_quadprog_script(self) -> str:
        """Get Octave script for quadratic programming."""
        
        return '''
% Quadratic Programming Solver for Octave
% Load problem data
addpath('{octave_dir}');
problem_data = loadjson('{problem_file}');

% Extract problem matrices
if ~isempty(problem_data.Q)
    Q = problem_data.Q;
else
    error('Quadratic programming requires Q matrix');
end

if ~isempty(problem_data.c)
    c = problem_data.c(:);
else
    c = zeros(size(Q, 1), 1);
end

% Combine constraints
A = [];
b = [];
Aeq = [];
beq = [];

if ~isempty(problem_data.A_ineq) && ~isempty(problem_data.b_ineq)
    A = problem_data.A_ineq;
    b = problem_data.b_ineq(:);
end

if ~isempty(problem_data.A_eq) && ~isempty(problem_data.b_eq)
    Aeq = problem_data.A_eq;
    beq = problem_data.b_eq(:);
end

% Set up bounds
lb = [];
ub = [];
if ~isempty(problem_data.bounds)
    n_vars = size(Q, 1);
    lb = -Inf(n_vars, 1);
    ub = Inf(n_vars, 1);
    
    for i = 1:length(problem_data.bounds)
        if ~isempty(problem_data.bounds{{i}})
            if length(problem_data.bounds{{i}}) >= 1 && ~isempty(problem_data.bounds{{i}}{{1}})
                lb(i) = problem_data.bounds{{i}}{{1}};
            end
            if length(problem_data.bounds{{i}}) >= 2 && ~isempty(problem_data.bounds{{i}}{{2}})
                ub(i) = problem_data.bounds{{i}}{{2}};
            end
        end
    end
end

% Solve using quadprog (if available) or alternative
try
    if exist('quadprog', 'file')
        % Use Octave's quadprog if available
        [x, fval, exitflag, output] = quadprog(Q, c, A, b, Aeq, beq, lb, ub);
        
        % Map exit flags to status
        if exitflag > 0
            status = 'optimal';
        elseif exitflag == 0
            status = 'max_iter';
        elseif exitflag == -2
            status = 'infeasible';
        elseif exitflag == -3
            status = 'unbounded';
        else
            status = 'error';
        end
        
        iterations = [];
        if isfield(output, 'iterations')
            iterations = output.iterations;
        end
        
    else
        % Fallback to basic QP solver
        warning('quadprog not available, using basic solver');
        
        % Simple unconstrained QP: min 0.5*x'*Q*x + c'*x
        if isempty(A) && isempty(Aeq)
            % Unconstrained case
            if all(eig(Q) > 1e-12)  % Positive definite
                x = -Q \\ c;
                fval = 0.5 * x' * Q * x + c' * x;
                status = 'optimal';
                iterations = 1;
            else
                status = 'error';
                x = [];
                fval = NaN;
                iterations = 0;
            end
        else
            % Constrained case - simplified solver
            status = 'error';
            x = [];
            fval = NaN;
            iterations = 0;
        end
    end
    
    % Prepare result data
    result.status = status;
    result.objective_value = fval;
    result.iterations = iterations;
    result.solution = x;
    
catch err
    result.status = 'error';
    result.error = err.message;
    result.objective_value = NaN;
    result.iterations = 0;
end

% Save results
savejson(result, '{result_file}');
'''

    def get_version(self) -> str:
        """Get Octave version information."""
        
        try:
            result = subprocess.run(
                [self.octave_path, '--version'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                # Extract version from first line
                first_line = result.stdout.split('\n')[0]
                return first_line.strip()
            else:
                return "Unknown Octave version"
                
        except Exception:
            return "Octave version unavailable"
    
    def is_available(self) -> bool:
        """Check if Octave is available and working."""
        return self._verify_octave()


def create_octave_solver(config: Optional[Dict] = None) -> OctaveSolver:
    """
    Factory function to create Octave solver instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OctaveSolver instance
    """
    config = config or {}
    
    return OctaveSolver(
        name=config.get('name', 'Octave'),
        timeout=config.get('timeout', 300.0),
        octave_path=config.get('octave_path'),
        options=config.get('options', {})
    )