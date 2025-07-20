# Optimization Solver Benchmark System - Technical Design

Detailed technical specifications for the optimization solver benchmark system supporting Python and MATLAB/Octave solvers across LP, QP, SOCP, and SDP problems.

---

## System Overview

**Purpose**: Benchmark optimization solvers using external problem libraries (DIMACS, SDPLIB) with minimal configuration for unbiased performance evaluation.

**Core Components**:
- **Problem Loaders**: Parse MAT/DAT formats → standardized ProblemData
- **Solver Interfaces**: Execute Python/MATLAB solvers → standardized SolverResult  
- **Database Storage**: SQLite with complete version tracking
- **Report Generation**: HTML dashboards with CSV/JSON export

**Supported Systems**:
- **Python Solvers (9)**: SciPy, CVXPY backends (CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HIGHS)
- **MATLAB/Octave Solvers (2)**: SeDuMi, SDPT3
- **Problem Types**: LP, QP, SOCP, SDP (139 problems total)

---

## Project Structure

### Directory Layout
```
optimization-solver-benchmark/
├── main.py                     # Entry point with argument parsing
├── config/                     # Configuration files
│   ├── site_config.yaml        # Site metadata and overview
│   └── problem_registry.yaml   # Problem metadata and file paths
├── scripts/
│   ├── benchmark/              # Benchmark execution engine
│   │   ├── __init__.py
│   │   └── runner.py           # Main BenchmarkRunner class
│   ├── solvers/                # Solver interface implementations
│   │   ├── __init__.py
│   │   ├── solver_interface.py # Abstract base classes
│   │   ├── python/             # Python solver implementations
│   │   │   ├── __init__.py
│   │   │   ├── python_interface.py   # Python solver coordinator
│   │   │   ├── cvxpy_runner.py       # CVXPY backend handler
│   │   │   └── scipy_runner.py       # SciPy linprog handler
│   │   └── matlab_octave/      # MATLAB/Octave integration
│   │       ├── __init__.py
│   │       ├── matlab_interface.py   # Python-MATLAB bridge
│   │       ├── matlab_interface.m    # MATLAB entry point
│   │       ├── sedumi_runner.m       # SeDuMi solver wrapper
│   │       ├── sdpt3_runner.m        # SDPT3 solver wrapper
│   │       ├── setup_matlab_solvers.m # MEX compilation script
│   │       ├── sedumi/         # SeDuMi solver (git submodule)
│   │       └── sdpt3/          # SDPT3 solver (git submodule)
│   ├── data_loaders/           # Problem format loaders
│   │   ├── __init__.py
│   │   ├── problem_loader.py   # ProblemData class definition
│   │   ├── python/             # Python format loaders
│   │   │   ├── __init__.py
│   │   │   ├── problem_interface.py  # Problem loading coordinator
│   │   │   ├── mat_loader.py         # DIMACS .mat loader
│   │   │   └── dat_loader.py         # SDPLIB .dat-s loader
│   │   └── matlab_octave/      # MATLAB format loaders
│   │       ├── mat_loader.m    # MATLAB .mat loader
│   │       └── dat_loader.m    # MATLAB .dat-s loader
│   ├── database/               # Database management
│   │   ├── __init__.py
│   │   ├── database_manager.py # Database operations
│   │   ├── models.py           # Data models
│   │   └── schema.sql          # Database schema
│   ├── reporting/              # Report generation
│   │   ├── __init__.py
│   │   ├── html_generator.py   # HTML report creation
│   │   ├── result_processor.py # Result aggregation
│   │   └── data_exporter.py    # JSON/CSV export
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       ├── environment_info.py # System information capture
│       ├── git_utils.py        # Git operations
│       ├── logger.py           # Logging configuration
│       └── temp_file_manager.py # Temporary file handling
├── problems/                   # Problem library files
│   ├── DIMACS/                 # External DIMACS library (git submodule)
│   └── SDPLIB/                 # External SDPLIB library (git submodule)
├── database/                   # SQLite database files
│   └── results.db              # Benchmark results storage
├── docs/                       # Generated reports and documentation
│   ├── pages/                  # Generated HTML reports
│   │   ├── index.html          # Main dashboard
│   │   ├── results_matrix.html # Problems × Solvers matrix
│   │   ├── raw_data.html       # Detailed data view
│   │   └── data/               # Exported data files
│   │       ├── benchmark_results.json
│   │       ├── benchmark_results.csv
│   │       └── summary.json
│   └── development/            # Development documentation
└── requirements.txt            # Python dependencies
```

---

## Component Architecture

### System Data Flow
```
Problem Files → Loaders → ProblemData → Solver Interfaces → SolverResult → Database → Reports
     ↓              ↓           ↓              ↓              ↓           ↓         ↓
DIMACS (.mat)   MATLoader   Unified      Python/MATLAB   Standardized SQLite  HTML/CSV
SDPLIB (.dat-s) DATLoader   Format       Execution       Results     Storage   Export
```

### Component Hierarchy
```
BenchmarkRunner (scripts/benchmark/runner.py)
├── ProblemInterface (scripts/data_loaders/python/problem_interface.py)
│   ├── MATLoader (scripts/data_loaders/python/mat_loader.py)
│   └── DATLoader (scripts/data_loaders/python/dat_loader.py)
├── PythonInterface (scripts/solvers/python/python_interface.py)
│   ├── CvxpyRunner (scripts/solvers/python/cvxpy_runner.py)
│   └── ScipyRunner (scripts/solvers/python/scipy_runner.py)
├── MatlabInterface (scripts/solvers/matlab_octave/matlab_interface.py)
│   ├── matlab_interface.m → sedumi_runner.m
│   └── matlab_interface.m → sdpt3_runner.m
└── DatabaseManager (scripts/database/database_manager.py)
    └── ResultProcessor (scripts/reporting/result_processor.py)
```

---

## Core Data Models

### Problem Data Structure
```python
# scripts/data_loaders/problem_loader.py
class ProblemData:
    """Unified problem representation using SeDuMi format"""
    def __init__(self, name: str, problem_class: str):
        self.name = name                     # Problem identifier
        self.problem_class = problem_class   # 'LP', 'QP', 'SOCP', 'SDP'
        
        # SeDuMi standard format (internal representation)
        self.A_eq = None          # Constraint matrix (sparse)
        self.b_eq = None          # RHS vector
        self.c = None             # Objective coefficients
        self.cone_structure = {}  # Cone constraints specification
        
        # Optional QP data
        self.P = None             # Quadratic term matrix
        
        # Problem metadata
        self._num_variables = 0
        self._num_constraints = 0
        self.metadata = {}        # Additional problem information
```

### Solver Result Structure
```python
# scripts/solvers/solver_interface.py
@dataclass
class SolverResult:
    """Standardized result format for all solvers"""
    solve_time: float                        # Execution time in seconds
    status: str                              # 'optimal', 'infeasible', 'unbounded', 'error'
    primal_objective_value: Optional[float] = None
    dual_objective_value: Optional[float] = None
    duality_gap: Optional[float] = None
    primal_infeasibility: Optional[float] = None
    dual_infeasibility: Optional[float] = None
    iterations: Optional[int] = None
    solver_name: Optional[str] = None
    solver_version: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_error_result(cls, error_msg: str, solve_time: float = 0.0) -> 'SolverResult':
        """Create standardized error result"""
        return cls(solve_time=solve_time, status='error', 
                  additional_info={'error_message': error_msg})
```

---

## Implementation Details

### 1. Problem Loading Implementation

#### Configuration Registry
```yaml
# config/problem_registry.yaml
problem_libraries:
  nb:  # DIMACS problem example
    display_name: "NB (DIMACS)"
    file_path: "problems/DIMACS/data/ANTENNA/nb.mat.gz"
    file_type: "mat"
    library_name: "DIMACS"
    for_test_flag: true
    
  arch0:  # SDPLIB problem example
    display_name: "ARCH0 (SDPLIB)"
    file_path: "problems/SDPLIB/data/arch0.dat-s"
    file_type: "dat-s"
    library_name: "SDPLIB"
    for_test_flag: true
```

#### Problem Interface Implementation
```python
# scripts/data_loaders/python/problem_interface.py
class ProblemInterface:
    """Centralized problem loading and registry management"""
    
    FORMAT_LOADERS = {
        "mat": MATLoader,        # SeDuMi format (DIMACS)
        "dat-s": DATLoader,      # SDPA format (SDPLIB) 
        # Future extensions for MPS, QPS, Python formats
    }
    
    def load_problem(self, problem_name: str) -> ProblemData:
        """Load problem using appropriate format loader"""
        config = self.get_problem_config(problem_name)
        file_type = config['file_type']
        file_path = config['file_path']
        
        loader_class = self.FORMAT_LOADERS[file_type]
        loader = loader_class()
        return loader.load(problem_name, file_path)
        
    def get_problem_config(self, problem_name: str) -> Dict[str, Any]:
        """Get problem configuration from registry"""
        # Load and parse problem_registry.yaml
        # Return configuration for specified problem
        
    def get_available_problems(self, library_filter: List[str] = None) -> List[str]:
        """Get filtered list of available problems"""
        # Filter by library name if specified
```

### 2. Python Solver Integration

#### Solver Configuration
```python
# scripts/solvers/python/python_interface.py
PYTHON_SOLVER_CONFIGS = {
    'cvxpy_clarabel': {
        'solver_backend': 'CLARABEL',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['LP', 'QP', 'SOCP', 'SDP']
    },
    'cvxpy_scs': {
        'solver_backend': 'SCS',
        'runner_class': 'CvxpyRunner', 
        'problem_types': ['LP', 'QP', 'SOCP', 'SDP']
    },
    'cvxpy_ecos': {
        'solver_backend': 'ECOS',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['LP', 'QP', 'SOCP']
    },
    'cvxpy_osqp': {
        'solver_backend': 'OSQP',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['QP']
    },
    'cvxpy_cvxopt': {
        'solver_backend': 'CVXOPT',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['LP', 'QP', 'SOCP', 'SDP']
    },
    'cvxpy_sdpa': {
        'solver_backend': 'SDPA',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['SDP']
    },
    'cvxpy_scip': {
        'solver_backend': 'SCIP',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['LP', 'QP']
    },
    'cvxpy_highs': {
        'solver_backend': 'HIGHS',
        'runner_class': 'CvxpyRunner',
        'problem_types': ['LP']
    },
    'scipy_linprog': {
        'method': 'highs',
        'runner_class': 'ScipyRunner',
        'problem_types': ['LP']
    },
}
```

#### Python Execution Flow
```python
# scripts/solvers/python/python_interface.py
class PythonInterface:
    def solve(self, problem_name: str, solver_name: str, 
              problem_data: Optional[ProblemData] = None,
              timeout: Optional[float] = None) -> SolverResult:
        """Execute Python solver with unified interface"""
        
        # 1. Load problem data if not provided
        if problem_data is None:
            problem_data = self.problem_interface.load_problem(problem_name)
        
        # 2. Get solver configuration
        solver_config = self.PYTHON_SOLVER_CONFIGS[solver_name]
        runner_class_name = solver_config['runner_class']
        
        # 3. Create appropriate runner
        if runner_class_name == 'CvxpyRunner':
            runner = CvxpyRunner(solver_config['solver_backend'])
        elif runner_class_name == 'ScipyRunner':
            runner = ScipyRunner(solver_config['method'])
        
        # 4. Execute solver
        result = runner.solve(problem_data, timeout=timeout)
        
        # 5. Add problem class information for database storage
        if not result.additional_info:
            result.additional_info = {}
        result.additional_info['problem_class'] = problem_data.problem_class
        
        return result
```

### 3. MATLAB Solver Integration

#### MATLAB Configuration
```python
# scripts/solvers/matlab_octave/matlab_interface.py
MATLAB_SOLVER_CONFIGS = {
    'matlab_sedumi': {
        'matlab_solver': 'sedumi',
        'runner_function': 'sedumi_runner'
    },
    'matlab_sdpt3': {
        'matlab_solver': 'sdpt3',
        'runner_function': 'sdpt3_runner'
    }
}
```

#### MATLAB Execution Flow
```python
# scripts/solvers/matlab_octave/matlab_interface.py
class MatlabInterface:
    def solve(self, problem_name: str, solver_name: str,
              problem_data: Optional[ProblemData] = None,
              timeout: Optional[float] = None) -> SolverResult:
        """Execute MATLAB solver via subprocess"""
        
        # 1. Validate solver configuration
        solver_config = self.MATLAB_SOLVER_CONFIGS[solver_name]
        
        # 2. Create temporary result file
        result_file = self.temp_manager.create_temp_file('.json')
        
        # 3. Construct MATLAB command
        matlab_command = f"""
        cd('{os.getcwd()}');
        addpath(genpath('scripts/solvers/matlab_octave'));
        matlab_interface('{problem_name}', '{solver_config["matlab_solver"]}', 
                        '{result_file}', false, '{solver_config["runner_function"]}');
        """
        
        # 4. Execute subprocess with timeout
        try:
            process = subprocess.run([
                'octave', '--eval', matlab_command
            ], timeout=timeout, capture_output=True, text=True)
            
            if process.returncode != 0:
                return SolverResult.create_error_result(f"MATLAB execution failed: {process.stderr}")
            
            # 5. Read JSON result
            with open(result_file, 'r') as f:
                matlab_result = json.load(f)
            
            # 6. Convert to SolverResult
            result = self._convert_matlab_result(matlab_result)
            
        except subprocess.TimeoutExpired:
            result = SolverResult.create_timeout_result(timeout)
        except Exception as e:
            result = SolverResult.create_error_result(str(e))
        finally:
            self.temp_manager.cleanup_temp_file(result_file)
        
        return result
```

#### MATLAB Interface Entry Point
```matlab
% scripts/solvers/matlab_octave/matlab_interface.m
function matlab_interface(problem_name, solver_name, result_file, save_solutions, runner_function)
    % Main MATLAB interface for benchmark execution
    
    try
        % 1. Load problem registry and resolve file path
        [problem_config, file_path] = read_problem_registry(problem_name);
        file_type = problem_config.file_type;
        
        % 2. Load problem data using appropriate loader
        if strcmp(file_type, 'mat')
            [A, b, c, K] = mat_loader(file_path);
        elseif strcmp(file_type, 'dat-s')
            [A, b, c, K] = dat_loader(file_path);
        else
            error('Unsupported file type: %s', file_type);
        end
        
        % 3. Execute solver using dynamic function call
        [x, y, result] = feval(runner_function, A, b, c, K);
        
        % 4. Calculate standardized metrics
        if ~isempty(x) && ~isempty(y)
            result = calculate_solver_metrics(result, x, y, A, b, c, K);
        end
        
        % 5. Convert to JSON and save
        json_result = convert_to_json_result(result);
        save_json_safely(json_result, result_file);
        
    catch ME
        error_result = create_error_result(ME, solver_name);
        save_json_safely(error_result, result_file);
    end
end
```

---

## Database Implementation

### Schema Definition
```sql
-- scripts/database/schema.sql
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Solver information
    solver_name TEXT NOT NULL,           -- 'cvxpy_clarabel', 'matlab_sedumi', etc.
    solver_version TEXT NOT NULL,        -- Full version with backend info
    
    -- Problem information  
    problem_library TEXT NOT NULL,       -- 'DIMACS', 'SDPLIB', 'internal'
    problem_name TEXT NOT NULL,          -- Problem identifier
    problem_type TEXT NOT NULL,          -- 'LP', 'QP', 'SOCP', 'SDP'
    
    -- Environment and execution context
    environment_info TEXT NOT NULL,      -- JSON string with system info
    commit_hash TEXT NOT NULL,           -- Git commit hash
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Standardized solver results
    solve_time REAL,                     -- Execution time in seconds
    status TEXT,                         -- 'OPTIMAL', 'INFEASIBLE', 'UNBOUNDED', 'ERROR', 'TIMEOUT', 'UNSUPPORTED'
    primal_objective_value REAL,        -- Primal objective value
    dual_objective_value REAL,          -- Dual objective value
    duality_gap REAL,                   -- Primal-dual gap
    primal_infeasibility REAL,          -- Primal constraint violation
    dual_infeasibility REAL,            -- Dual constraint violation
    iterations INTEGER,                  -- Number of solver iterations
    memo TEXT,                          -- Additional solver-specific info (JSON)
    
    UNIQUE(solver_name, solver_version, problem_library, problem_name, commit_hash, timestamp)
);

-- Indexes for efficient querying
CREATE INDEX idx_latest_results ON results(commit_hash, environment_info, timestamp DESC);
CREATE INDEX idx_solver_problem ON results(solver_name, problem_name);
CREATE INDEX idx_problem_type ON results(problem_type);
```

### Database Manager Implementation
```python
# scripts/database/database_manager.py
class DatabaseManager:
    def store_result(self, solver_name: str, problem_name: str, 
                    result: SolverResult, problem_config: Dict[str, Any]) -> None:
        """Store benchmark result with complete metadata"""
        
        # Extract metadata
        problem_library = problem_config.get('library_name', 'internal')
        problem_type = result.additional_info.get('problem_class', 'UNKNOWN')
        
        # Get environment and version information
        environment_info = self.environment_collector.get_environment_info()
        commit_hash = self.git_utils.get_current_commit_hash()
        
        # Insert into database
        query = """
        INSERT OR REPLACE INTO results 
        (solver_name, solver_version, problem_library, problem_name, problem_type,
         environment_info, commit_hash, solve_time, status, primal_objective_value,
         dual_objective_value, duality_gap, primal_infeasibility, dual_infeasibility,
         iterations, memo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.cursor.execute(query, (
            solver_name, result.solver_version, problem_library, problem_name, problem_type,
            json.dumps(environment_info), commit_hash, result.solve_time, result.status,
            result.primal_objective_value, result.dual_objective_value, result.duality_gap,
            result.primal_infeasibility, result.dual_infeasibility, result.iterations,
            json.dumps(result.additional_info) if result.additional_info else None
        ))
        
        self.connection.commit()
```

---

## Report Generation

### HTML Generation Implementation
```python
# scripts/reporting/html_generator.py
class HTMLGenerator:
    def generate_all_reports(self) -> None:
        """Generate complete set of HTML reports"""
        
        # Load latest results from database
        results = self.db.get_latest_results()
        
        # Generate main dashboard
        self.generate_dashboard(results, 'docs/pages/index.html')
        
        # Generate results matrix
        self.generate_results_matrix(results, 'docs/pages/results_matrix.html')
        
        # Generate raw data view
        self.generate_raw_data_view(results, 'docs/pages/raw_data.html')
        
        # Export data files
        self.data_exporter.export_json(results, 'docs/pages/data/benchmark_results.json')
        self.data_exporter.export_csv(results, 'docs/pages/data/benchmark_results.csv')
        self.data_exporter.export_summary(results, 'docs/pages/data/summary.json')
```

---

## Fair Benchmarking Implementation

### Minimal Configuration Approach
```python
# Solver configurations use minimal parameters to ensure fair comparison
FAIR_BENCHMARKING_CONFIG = {
    'python_solvers': {
        'cvxpy_options': {'verbose': False},  # Only suppress output
        'scipy_options': {'method': 'highs'}  # Use default method
    },
    'matlab_solvers': {
        'sedumi_options': {'fid': 0},         # Only suppress output
        'sdpt3_options': {'printlevel': 0}    # Only suppress output
    }
}
```

### Reproducibility Implementation
```python
# scripts/utils/environment_info.py
class EnvironmentCollector:
    def get_environment_info(self) -> Dict[str, Any]:
        """Collect complete environment information for reproducibility"""
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'installed_packages': self._get_package_versions(),
            'git_commit': self.git_utils.get_current_commit_hash(),
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname()
        }
```

---

## Development Guidelines

### Adding New Python Solvers
1. **Add configuration** to `PYTHON_SOLVER_CONFIGS` in `python_interface.py`
2. **Create runner class** if needed (or extend existing `CvxpyRunner`)
3. **Update problem type compatibility** in configuration
4. **Test with validation framework**

### Adding New MATLAB Solvers
1. **Create solver runner** `{solver}_runner.m` following standard interface
2. **Add configuration** to `MATLAB_SOLVER_CONFIGS` in `matlab_interface.py`
3. **Implement MEX compilation** in `setup_matlab_solvers.m`
4. **Test with validation framework**

### Adding New Problem Formats
1. **Create loader classes** in both `python/` and `matlab_octave/` directories
2. **Add format mapping** to `FORMAT_LOADERS` in `problem_interface.py`
3. **Update problem registry** to include new format problems
4. **Implement format conversion** to unified SeDuMi format

---

This detailed design provides comprehensive implementation guidance while maintaining focus on research applications and eliminating unnecessary production complexity.