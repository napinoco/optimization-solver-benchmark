# Optimization Solver Benchmark System - Re-Architected Design

This document provides technical specifications for the **re-architected** optimization solver benchmark system focused on simplicity, reliability, and maintainability.

---

## System Architecture

### Simplified Data Flow Architecture
```
LOCAL DEVELOPMENT:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Problem   │───▶│   Solver     │───▶│   Result    │
│   Loading   │    │   Execution  │    │  Collection │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Validation │    │  Environment │    │  Database   │
│   & Caching │    │    Capture   │    │   Storage   │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
                                     ┌─────────────┐
                                     │   Report    │
                                     │ Generation  │
                                     └─────────────┘
                                              │
                                              ▼
                                     ┌─────────────┐
                                     │  Commit to  │
                                     │   docs/     │
                                     └─────────────┘

GITHUB ACTIONS (Publishing Only):
┌─────────────┐    ┌──────────────┐
│  Pre-built  │───▶│   GitHub     │
│    docs/    │    │    Pages     │
└─────────────┘    └──────────────┘
```

### Component Interaction (Simplified)
```
LOCAL DEVELOPMENT:
├── Problem Loading (Local files only)
├── Python Solver Execution (CVXPY + SciPy)
├── Result Collection & Database Storage
├── HTML Report Generation
├── Data Export (JSON/CSV)
└── Commit Generated Files

GITHUB ACTIONS (Minimal):
├── PR Preview (Lightweight benchmark + publish)
└── Main Branch Publishing (Static file deployment only)
```

---

## Re-Architected Directory Structure

```
optimization-solver-benchmark/
├── README.md                    # Project overview and quick start (updated)
├── CLAUDE.md                    # AI assistant integration context
├── LICENSE                      # MIT license
├── requirements.txt             # All dependencies consolidated
│
├── .github/                     # GitHub Actions workflows
│   └── workflows/
│       ├── deploy.yml             # Deploy reports and PR previews
│       └── validate.yml           # Codebase validation (no benchmarking)
│
├── config/                      # Configuration files
│   ├── site_config.yaml          # Site display information
│   ├── solver_registry.yaml      # Available solvers list
│   └── problem_registry.yaml     # Available problems list (moved from problems/)
│
├── problems/                    # Problem instances
│   ├── light_set/               # Internal problems
│   │   ├── lp/                  # Linear programming (.mps files)
│   │   ├── qp/                  # Quadratic programming (.qps files)
│   │   ├── socp/                # Second-order cone programming (Python)
│   │   └── sdp/                 # Semidefinite programming (Python)
│   ├── DIMACS/                  # External DIMACS library
│   └── SDPLIB/                  # External SDPLIB library
│
├── scripts/
│   ├── benchmark/               # Benchmark execution and database storage
│   │   ├── __init__.py
│   │   ├── runner.py             # Main benchmark execution logic
│   │   ├── database_manager.py   # Database operations and result storage
│   │   └── environment_info.py   # Environment information gathering
│   │
│   ├── data_loaders/            # ETL: Data loading and format conversion
│   │   ├── __init__.py
│   │   ├── python/               # Python-based loaders
│   │   │   ├── __init__.py
│   │   │   ├── mat_loader.py     # DIMACS .mat file loader
│   │   │   ├── dat_loader.py     # SDPLIB .dat-s file loader
│   │   │   ├── mps_loader.py     # MPS format loader
│   │   │   ├── qps_loader.py     # QPS format loader
│   │   │   ├── python_loader.py  # Python problem loader
│   │   │   └── cvxpy_converter.py # Convert to CVXPY format
│   │   └── matlab_octave/        # MATLAB/Octave loaders (future)
│   │       └── .gitkeep
│   │
│   ├── solvers/                 # Solver execution with standardized output
│   │   ├── __init__.py
│   │   ├── solver_interface.py   # Abstract solver interface
│   │   ├── python/               # Python solvers
│   │   │   ├── __init__.py
│   │   │   ├── cvxpy_runner.py   # CVXPY solver implementation
│   │   │   └── scipy_runner.py   # SciPy solver implementation
│   │   └── matlab_octave/        # MATLAB/Octave solvers (future)
│   │       └── .gitkeep
│   │
│   ├── reporting/               # HTML generation and data extraction
│   │   ├── __init__.py
│   │   ├── html_generator.py     # Generate overview, results_matrix, raw_data
│   │   ├── data_exporter.py      # Export JSON/CSV data
│   │   └── result_processor.py   # Process latest results from database
│   │
│   └── database/                # Database models and operations
│       ├── __init__.py
│       ├── models.py             # Single denormalized table model
│       └── schema.sql            # Database schema definition
│
├── docs/                        # GitHub Pages output (published data)
│   ├── pages/                   # Generated HTML and data
│   │   ├── index.html           # Overview report
│   │   ├── results_matrix.html  # Problems × solvers matrix
│   │   ├── raw_data.html        # Raw data display
│   │   ├── assets/              # CSS, JS, images
│   │   └── data/                # JSON/CSV exports
│   ├── development/             # Developer documentation
│   └── guides/                  # User documentation
│
├── database/                    # SQLite database
│   └── results.db               # Single denormalized results table
│
├── logs/                        # Log files (local only, .gitignore)
│   └── benchmark.log            # Structured execution logs
│
└── tests/                       # Test suite
    ├── unit/                    # Unit tests
    ├── integration/             # Integration tests
    └── fixtures/                # Test data and configurations
```

---

## Re-Architected Core Components

### 1. GitHub Actions Workflows (Minimal)

#### deploy.yml - Unified Deployment System
```yaml
# Key features:
# - Triggers on push to main branch and pull requests
# - Publishes PRE-BUILT docs/ folder to GitHub Pages
# - No benchmark execution in CI
# - Unified workflow for both production and PR previews
# - Auto-deploy PR previews to gh-pages/pr-preview/pr-{number}/
# - Auto-cleanup when PR closed
# - Preview banners and metadata injection
# - Comments with preview URLs on PRs
# - Preserves PR preview subdirectories
```

#### validate.yml - Codebase Validation Only  
```yaml
# Key features:
# - Lightweight CI validation without benchmarking
# - Validates configuration files can be loaded
# - Checks Python dependencies installation
# - Verifies core system components initialize correctly
# - Tests solver backend availability
# - No benchmark execution or report generation
# - Fast validation for development workflow
```

### 2. Configuration Management (Re-architected)

The new configuration structure eliminates `benchmark_config.yaml` and consolidates all configuration into three focused files. The system now derives configuration directly from the registries and uses sensible defaults.

#### config/site_config.yaml - Site Display Information
```yaml
site:
  title: "Optimization Solver Benchmark"
  author: "Your Name"
  description: "Benchmarking optimization solvers with fair comparison"
  url: "https://your-username.github.io/optimization-solver-benchmark"

github:
  username: "your-username"
  repository: "optimization-solver-benchmark"
```

#### config/solver_registry.yaml - Solver Display Names Only
```yaml
# Simplified solver registry - only display names for reporting
# Actual solver initialization logic is in code for better maintainability
solvers:
  scipy_linprog:
    display_name: "SciPy linprog"
    
  cvxpy_clarabel:
    display_name: "CLARABEL (via CVXPY)"
    
  cvxpy_scs:
    display_name: "SCS (via CVXPY)"
    
  cvxpy_ecos:
    display_name: "ECOS (via CVXPY)"
    
  cvxpy_osqp:
    display_name: "OSQP (via CVXPY)"
```

This simplified approach moves solver initialization logic to code while maintaining clean display names for reports. The solver selection logic becomes:

```python
# In BenchmarkRunner.create_solver()
def create_solver(self, solver_name: str) -> SolverInterface:
    """Create solver instance based on solver name"""
    
    if solver_name == "scipy_linprog":
        return SciPySolver()
    elif solver_name == "cvxpy_clarabel":
        return CVXPYSolver(backend="CLARABEL")
    elif solver_name == "cvxpy_scs":
        return CVXPYSolver(backend="SCS")
    elif solver_name == "cvxpy_ecos":
        return CVXPYSolver(backend="ECOS")
    elif solver_name == "cvxpy_osqp":
        return CVXPYSolver(backend="OSQP")
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
```

#### config/problem_registry.yaml - Flat Problem Structure
```yaml
# Flat problem structure - each problem is a top-level entry
# Eliminates light_set in favor of real problems with test flags
problem_libraries:
  
  # Small-scale test problems from DIMACS/SDPLIB  
  nb:
    display_name: "ANTENNA NB (DIMACS)"
    file_path: "problems/DIMACS/data/ANTENNA/nb.mat.gz"
    file_type: "mat"
    problem_type: "SDP"
    library_name: "DIMACS"
    for_test_flag: true  # Small problem suitable for testing
    known_objective_value: -12.8  # Known optimal value for validation
    
  arch0:
    display_name: "ARCH0 (SDPLIB)"
    file_path: "problems/SDPLIB/data/arch0.dat-s"
    file_type: "dat-s" 
    problem_type: "SDP"
    library_name: "SDPLIB"
    for_test_flag: true  # Small problem suitable for testing
    known_objective_value: -5.6506  # Known optimal value
    
  # Larger production problems
  hinf12:
    display_name: "H-infinity Control 12 (DIMACS)"
    file_path: "problems/DIMACS/data/HINF/hinf12.mat.gz"
    file_type: "mat"
    problem_type: "SDP"
    library_name: "DIMACS"
    for_test_flag: false
    # known_objective_value: null  # Unknown - omit field
    
  control1:
    display_name: "Control Problem 1 (SDPLIB)"
    file_path: "problems/SDPLIB/data/control1.dat-s"
    file_type: "dat-s"
    problem_type: "SDP" 
    library_name: "SDPLIB"
    for_test_flag: false
    known_objective_value: 20.8  # Known optimal value
    
  gpp100:
    display_name: "Graph Partitioning 100 (SDPLIB)"
    file_path: "problems/SDPLIB/data/gpp100.dat-s"
    file_type: "dat-s"
    problem_type: "SDP"
    library_name: "SDPLIB" 
    for_test_flag: false
    # known_objective_value: null  # Unknown - omit field
    
  # Linear programming problems
  simple_lp_test:
    display_name: "Simple LP Test (Internal)"
    file_path: "problems/light_set/lp/simple_lp.mps"
    file_type: "mps"
    problem_type: "LP"
    library_name: "internal"
    for_test_flag: true  # Synthetic test problem
    known_objective_value: 5.0  # Known optimal value
```

This structure provides:
- **Flat hierarchy**: Direct problem access without nested library structure
- **Test problem identification**: `for_test_flag` to identify small problems for quick testing
- **Known objective values**: Optional field for result validation
- **Library attribution**: Clear source library tracking
- **Real problems for testing**: Eliminates synthetic light_set in favor of real small problems

### 3. Re-architected Benchmark Execution

The new execution system removes complex backend selection and aggregation, focusing on simple, direct execution with standardized results storage.

#### scripts/benchmark/runner.py - Main Execution Logic
```python
class BenchmarkRunner:
    """Simplified benchmark execution with direct database storage"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.environment_info = self.gather_environment_info()
        self.commit_hash = self.get_git_commit_hash()
        
        # Load configurations
        self.solver_registry = self.load_solver_registry()
        self.problem_registry = self.load_problem_registry()
    
    def run_single_benchmark(self, problem_name: str, solver_name: str) -> None:
        """Execute single problem-solver combination and store result"""
        
        # 1. Load problem using appropriate loader
        problem_config = self.problem_registry[problem_name]
        problem_data = self.load_problem(problem_name, problem_config)
        
        # 2. Initialize solver
        solver = self.create_solver(solver_name)
        
        # 3. Convert problem to solver format
        converter = CVXPYConverter()
        cvxpy_problem = converter.convert(problem_data)
        
        # 4. Execute solver with timeout
        try:
            start_time = time.time()
            result = solver.solve(cvxpy_problem)
            solve_time = time.time() - start_time
            
            # 5. Store standardized result in database
            self.store_result(solver_name, problem_name, result, solve_time)
            
        except Exception as e:
            # Store error result
            self.store_error_result(solver_name, problem_name, str(e))
    
    def run_benchmark_batch(self, problems: List[str], solvers: List[str]) -> None:
        """Run benchmark for all problem-solver combinations"""
        
        total_combinations = len(problems) * len(solvers)
        completed = 0
        
        for problem_name in problems:
            for solver_name in solvers:
                try:
                    logger.info(f"Running {solver_name} on {problem_name}")
                    self.run_single_benchmark(problem_name, solver_name)
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_combinations}")
                    
                except Exception as e:
                    logger.error(f"Failed {solver_name} on {problem_name}: {e}")
                    completed += 1
```

#### scripts/benchmark/database_manager.py - Database Operations
```python
class DatabaseManager:
    """Handles all database operations for benchmark results"""
    
    def __init__(self, db_path: str = "database/results.db"):
        self.db_path = db_path
        self.ensure_schema()
    
    def store_result(self, solver_name: str, solver_version: str, 
                    problem_library: str, problem_name: str, problem_type: str,
                    environment_info: str, commit_hash: str,
                    solve_time: float, status: str, 
                    primal_objective: float, dual_objective: float,
                    duality_gap: float, primal_infeas: float, dual_infeas: float,
                    iterations: int) -> None:
        """Store single benchmark result (append-only)"""
        
        # Insert into results table without initialization
        # Preserves all historical data
        
    def get_latest_results(self) -> List[BenchmarkResult]:
        """Get latest results for reporting"""
        
        # Query for results with latest commit_hash and environment_info
        # Use timestamp as tiebreaker for truly latest results
        query = """
        SELECT * FROM results 
        WHERE (commit_hash, environment_info, timestamp) IN (
            SELECT commit_hash, environment_info, MAX(timestamp)
            FROM results 
            GROUP BY solver_name, solver_version, problem_library, problem_name
        )
        ORDER BY problem_library, problem_name, solver_name
        """
        
    def get_solver_problem_history(self, solver_name: str, problem_name: str) -> List[BenchmarkResult]:
        """Get historical results for analysis"""
```

#### scripts/benchmark/environment_info.py - Environment Capture
```python
class EnvironmentInfo:
    """Capture and standardize environment information"""
    
    @staticmethod
    def gather() -> dict:
        """Collect comprehensive environment information"""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_cores": os.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "hostname": platform.node(),
            "user": getpass.getuser(),
            "timestamp": datetime.now().isoformat(),
            "timezone": str(datetime.now().astimezone().tzinfo)
        }
    
    @staticmethod 
    def get_git_commit_hash() -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
```

### 1. Database Architecture - Single Denormalized Table

#### Simplified Schema Design
```sql
-- Single denormalized results table with historical retention
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Solver information
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    
    -- Problem information  
    problem_library TEXT NOT NULL,        -- 'light_set', 'DIMACS', 'SDPLIB'
    problem_name TEXT NOT NULL,
    problem_type TEXT NOT NULL,           -- 'LP', 'QP', 'SOCP', 'SDP'
    
    -- Environment and execution context
    environment_info TEXT NOT NULL,      -- JSON string with system info
    commit_hash TEXT NOT NULL,           -- Git commit hash
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Standardized solver results
    solve_time REAL,                     -- Execution time in seconds
    status TEXT,                         -- 'optimal', 'infeasible', 'error', etc.
    primal_objective_value REAL,        -- Primal objective value
    dual_objective_value REAL,          -- Dual objective value (if available)
    duality_gap REAL,                   -- Duality gap
    primal_infeasibility REAL,          -- Primal infeasibility measure
    dual_infeasibility REAL,            -- Dual infeasibility measure
    iterations INTEGER,                  -- Number of solver iterations
    
    -- Unique constraint to prevent exact duplicates
    UNIQUE(solver_name, solver_version, problem_library, problem_name, commit_hash, timestamp)
);

-- Index for efficient latest results queries
CREATE INDEX idx_latest_results ON results(commit_hash, environment_info, timestamp DESC);
CREATE INDEX idx_solver_problem ON results(solver_name, problem_name);
```

#### Database Manager
```python
class DatabaseManager:
    """Manages database operations for benchmark results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def store_result(self, result: BenchmarkResult) -> None:
        """Store a single benchmark result (append-only)"""
        # No database initialization - preserve historical data
        
    def get_latest_results(self, commit_hash: str, environment_info: str) -> List[BenchmarkResult]:
        """Get latest results for specific commit and environment"""
        # Query for results with matching commit_hash and environment_info
        # Use timestamp as tiebreaker for truly latest results
        
    def get_solver_problem_history(self, solver_name: str, problem_name: str) -> List[BenchmarkResult]:
        """Get historical results for solver-problem combination"""
```

### 2. Data Loading and Format Conversion (ETL)

#### Format-Specific Loaders Architecture
```python
# Direct loader usage without dispatcher
class MATLoader:
    """Load DIMACS .mat files in SeDuMi format"""
    
    def load(self, file_path: str) -> ProblemData:
        """Load .mat file and extract problem matrices"""

class DATLoader:
    """Load SDPLIB .dat-s files in SDPA sparse format"""
    
    def load(self, file_path: str) -> ProblemData:
        """Parse SDPA sparse format and create problem data"""

class MPSLoader:
    """Load MPS format files (Linear Programming)"""
    
    def load(self, file_path: str) -> ProblemData:
        """Parse MPS format and create problem data"""

class QPSLoader:
    """Load QPS format files (Quadratic Programming)"""
    
    def load(self, file_path: str) -> ProblemData:
        """Parse QPS format and create problem data"""

class PythonLoader:
    """Load Python-defined problems"""
    
    def load(self, file_path: str) -> ProblemData:
        """Execute Python file and extract problem definition"""

# Unified conversion to solver format
class CVXPYConverter:
    """Convert any problem format to CVXPY representation"""
    
    def convert(self, problem_data: ProblemData) -> CVXPYProblem:
        """Convert problem to CVXPY format for solving"""
```

#### Loader Selection Logic
```python
# In BenchmarkRunner
def load_problem(self, problem_name: str, problem_library: str) -> ProblemData:
    """Load problem based on registry configuration"""
    
    # Get problem info from config/problem_registry.yaml
    problem_config = self.get_problem_config(problem_name, problem_library)
    file_type = problem_config['file_type']
    file_path = problem_config['file_path']
    
    # Select appropriate loader based on file type
    if file_type == 'mat':
        loader = MATLoader()
    elif file_type == 'dat-s':
        loader = DATLoader()
    elif file_type == 'mps':
        loader = MPSLoader()
    elif file_type == 'qps':
        loader = QPSLoader()
    elif file_type == 'python':
        loader = PythonLoader()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return loader.load(file_path)
```

### 3. Solver Architecture with Standardized Output

#### Solver Interface
```python
class SolverInterface:
    """Abstract interface for all solvers with standardized output"""
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """Solve problem and return standardized result"""
        
class SolverResult:
    """Standardized solver result format"""
    
    def __init__(self):
        self.solve_time: float
        self.status: str                      # 'optimal', 'infeasible', 'unbounded', 'error'
        self.primal_objective_value: float
        self.dual_objective_value: float      # Optional, may be None
        self.duality_gap: float              # Optional, may be None  
        self.primal_infeasibility: float     # Constraint violation measure
        self.dual_infeasibility: float       # Dual constraint violation measure
        self.iterations: int                 # Number of solver iterations
        self.solver_info: dict               # Additional solver-specific information
```

#### CVXPY and SciPy Solver Implementation
```python
class CVXPYSolver(SolverInterface):
    """CVXPY solver with multiple backend support"""
    
    def __init__(self, backend: str):
        self.backend = backend  # 'CLARABEL', 'SCS', 'ECOS', 'OSQP'
        self.solver_version = self.detect_version()
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """Solve using CVXPY with specified backend"""
        # Convert to CVXPY format
        # Solve with specified backend
        # Extract standardized results
        
    def detect_version(self) -> str:
        """Detect CVXPY and backend versions"""
        # Return format: "cvxpy-1.4.0+CLARABEL-0.6.0"

class SciPySolver(SolverInterface):
    """SciPy optimization solvers"""
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """Solve using appropriate SciPy method based on problem type"""
```

### 4. Benchmark Execution and Database Storage

#### Benchmark Runner
```python
class BenchmarkRunner:
    """Main benchmark execution with database storage"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.environment_info = self.gather_environment_info()
        self.commit_hash = self.get_git_commit_hash()
    
    def run_single_benchmark(self, problem_name: str, solver_name: str) -> None:
        """Run single problem-solver combination and store result"""
        # Load problem using appropriate loader
        # Execute solver 
        # Store result in database (append-only)
        
    def run_benchmark_batch(self, problems: List[str], solvers: List[str]) -> None:
        """Run benchmark for all problem-solver combinations"""
        # Loop through problems and solvers
        # Call run_single_benchmark for each combination
```

### 5. Simplified Reporting System  

#### HTML Report Generation
```python
class HTMLGenerator:
    """Generate simplified HTML reports"""
    
    def generate_overview(self) -> str:
        """Generate overview dashboard showing summary statistics"""
        
    def generate_results_matrix(self) -> str:
        """Generate problems × solvers results matrix"""
        
    def generate_raw_data(self) -> str:
        """Generate raw data table for detailed inspection"""

class DataExporter:
    """Export data in JSON and CSV formats"""
    
    def export_latest_results(self) -> None:
        """Export latest results to JSON and CSV files"""
        
class ResultProcessor:
    """Process latest results from database for reporting"""
    
    def get_latest_results_for_reporting(self) -> List[BenchmarkResult]:
        """Get latest results using commit_hash and environment_info with timestamp tiebreaker"""
```

---

## Requirements Management

### Consolidated requirements.txt
```
# Core dependencies
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
pyyaml>=6.0
sqlalchemy>=2.0.0
jinja2>=3.1.0

# Solver dependencies  
cvxpy>=1.4.0
clarabel>=0.6.0
scs>=3.2.0
ecos>=2.0.0
osqp>=0.6.0

# Development and testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Optional: file format support
h5py>=3.8.0           # For .mat file loading
```

---

## Main Execution Flow

### Command Line Interface
```python
# Main execution command structure
python main.py --benchmark --problems light_set --solvers scipy_linprog,cvxpy_clarabel
python main.py --benchmark --problems DIMACS --solvers cvxpy_scs
python main.py --report
```

### Execution Workflow
```
1. Configuration Loading
   ├── Load config/solver_registry.yaml
   ├── Load config/problem_registry.yaml  
   └── Initialize database connection

2. Problem and Solver Selection
   ├── Parse command line arguments
   ├── Filter problems by library/type
   └── Filter solvers by capability

3. Benchmark Execution
   ├── For each problem-solver combination:
   │   ├── Load problem using appropriate loader
   │   ├── Execute solver with standardized interface
   │   └── Store result in database (append-only)
   └── Continue execution despite individual failures

4. Report Generation
   ├── Query latest results from database
   ├── Generate HTML reports (overview, matrix, raw data)
   ├── Export JSON/CSV data
   └── Save to docs/pages/ directory
```

## Extension Points

### Adding New Solvers
1. **Implement SolverInterface**: Create new solver class following the interface
2. **Add to solver_registry.yaml**: Configure solver metadata and capabilities
3. **Update requirements.txt**: Add solver dependencies
4. **Test Integration**: Validate with existing problems

### Adding New Problem Libraries
1. **Create loader**: Implement format-specific loader in `scripts/data_loaders/`
2. **Update problem_registry.yaml**: Add library and problem metadata
3. **Test loading**: Ensure problems convert correctly to CVXPY format
4. **Validate results**: Check solver compatibility and result quality

### Adding New File Formats
1. **Implement loader**: Create format parser in `scripts/data_loaders/`
2. **Add converter**: Implement conversion to CVXPY representation
3. **Update dispatcher**: Add format mapping to ProblemLoader
4. **Test pipeline**: Validate end-to-end problem loading and solving

---

## Implementation Validation

### Testing Strategy
- **Unit Tests**: Individual component validation (data loaders, solvers, database operations)
- **Integration Tests**: Complete workflow testing (problem loading → solving → storage → reporting)
- **Format Tests**: Validate all problem file formats load correctly
- **Solver Tests**: Ensure all solvers produce standardized output format

### Error Handling
- **Graceful Degradation**: Continue benchmark execution despite individual solver/problem failures
- **Comprehensive Logging**: Structured logging with clear error messages and context
- **Timeout Management**: Respect solver timeout limits and handle hanging processes
- **Data Validation**: Validate solver results before database storage

### Security Considerations
- **Input Validation**: Validate problem file formats and configuration values
- **Resource Limits**: Enforce memory and execution time limits
- **Dependency Management**: Explicit version pinning for reproducible environments

---

*This re-architected design document provides technical specifications for the simplified, maintainable optimization solver benchmark system. For high-level concepts and project vision, see [basic_design.md](basic_design.md).*

---

## Migration and Implementation Plan

### Phase 1: Database and Configuration (Week 1)
1. **Database Restructuring**: Implement single denormalized results table
2. **Configuration Consolidation**: Move and restructure configuration files
3. **Requirements Consolidation**: Merge all requirements into single file

### Phase 2: Core Architecture (Week 2)  
4. **Data Loaders Implementation**: Create ETL system for all problem formats
5. **Solver Interface Standardization**: Implement standardized solver output format
6. **Database Manager**: Implement append-only database operations

### Phase 3: Benchmark and Reporting (Week 3)
7. **Benchmark Runner**: Implement problem-solver execution loop with database storage
8. **Simplified Reporting**: Generate three focused HTML reports (overview, matrix, raw data)
9. **Data Export**: JSON/CSV export functionality

### Phase 4: Integration and Testing (Week 4)
10. **End-to-End Testing**: Validate complete workflow from execution to reporting
11. **Performance Validation**: Ensure no significant performance regression
12. **Documentation Updates**: Update all documentation to reflect new architecture

---

*Last Updated: June 2025 - Re-Architecture Design Complete*

---

## Summary

This re-architected design focuses on:

1. **Simplicity**: Single denormalized database table, consolidated configuration files
2. **Maintainability**: Clear separation of concerns with dedicated ETL and solver modules  
3. **Reliability**: Append-only database, graceful error handling, comprehensive logging
4. **Extensibility**: Modular design for adding new solvers, problems, and file formats
5. **Clean Break**: Fresh start without backward compatibility constraints

### Key Benefits

- **Reduced Complexity**: Eliminates multi-table relationships and complex aggregation logic
- **Historical Preservation**: Append-only database maintains complete execution history
- **Fair Benchmarking**: Standardized solver interface ensures consistent result format
- **Easy Reporting**: Latest results query using commit_hash and environment_info with timestamp tiebreaker
- **Format Flexibility**: ETL system supports multiple problem file formats with unified conversion

This design provides a solid foundation for long-term development while addressing the complexity issues of the previous architecture.
