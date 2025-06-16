# Optimization Solver Benchmark System - Simplified Design

This document provides technical specifications for the **simplified** optimization solver benchmark system focused on reliability and maintainability.

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

## File and Directory Structure (Simplified)

```
optimization-solver-benchmark/
├── README.md                    # Project overview and quick start
├── architecture.md              # High-level system overview  
├── CLAUDE.md                    # AI assistant integration context
├── LICENSE                      # MIT license
│
├── .github/
│   └── workflows/
│       ├── deploy-pages.yml      # Publish pre-built docs/ to GitHub Pages
│       └── pr-preview.yml        # PR preview with lightweight benchmark
│
├── config/
│   ├── benchmark_config.yaml    # Local benchmark execution settings
│   ├── solvers.yaml              # Python solver definitions only
│   ├── site_config.yaml          # Author and site metadata
│   └── backend_templates/        # CVXPY backend configurations
│       ├── clarabel.yaml
│       ├── scs.yaml
│       ├── ecos.yaml
│       └── osqp.yaml
│
├── problems/
│   └── light_set/               # Local problems only
│       ├── lp/                  # Linear programming (.mps files)
│       ├── qp/                  # Quadratic programming (.qps files)
│       ├── socp/                # Second-order cone programming (Python)
│       └── sdp/                 # Semidefinite programming (Python)
│
├── scripts/
│   ├── benchmark/
│   │   ├── runner.py             # Benchmark execution engine
│   │   ├── solver_interface.py   # Solver abstraction layer
│   │   ├── problem_loader.py     # Problem loading (local only)
│   │   ├── result_collector.py   # Result collection and aggregation
│   │   ├── backend_selector.py   # CVXPY backend selection
│   │   └── environment_info.py   # Environment information gathering
│   │
│   ├── solvers/
│   │   └── python/               # Python solvers only
│   │       ├── cvxpy_runner.py   # CVXPY execution with multiple backends
│   │       └── scipy_runner.py   # SciPy optimization suite
│   │
│   ├── database/
│   │   ├── models.py             # SQLAlchemy data models
│   │   └── classification_storage.py # Problem classification storage
│   │
│   ├── reporting/
│   │   ├── simple_html_generator.py  # Simple HTML report generation
│   │   ├── data_publisher.py     # JSON/CSV data publishing
│   │   ├── data_validator.py     # Result validation
│   │   ├── export.py             # Multi-format data export
│   │   └── statistics.py         # Statistical calculations
│   │
│   ├── analytics/
│   │   ├── statistical_analysis.py  # Statistical analysis
│   │   ├── performance_profiler.py  # Performance profiling
│   │   └── analytics_runner.py   # Analytics execution
│   │
│   └── utils/
│       ├── config_loader.py      # Configuration loading
│       ├── logger.py             # Structured logging
│       ├── validation.py         # Data validation utilities
│       ├── problem_classifier.py # Problem type classification
│       └── solver_diagnostics.py # Solver capability detection
│
├── docs/                        # GitHub Pages output (PRE-BUILT)
│   ├── index.html               # Main dashboard (committed)
│   ├── solver_comparison.html   # Solver performance comparison (committed)
│   ├── problem_analysis.html    # Problem-wise analysis (committed)
│   ├── results_matrix.html      # Problems × solvers matrix (committed)
│   ├── statistical_analysis.html # Statistical analysis report (committed)
│   ├── performance_profiling.html # Performance profiling report (committed)
│   ├── environment_info.html    # Environment information (committed)
│   ├── assets/
│   │   ├── css/style.css        # Custom styles (committed)
│   │   └── js/                  # JavaScript for interactivity (committed)
│   └── data/
│       ├── results.json         # Complete benchmark results (committed)
│       ├── summary.json         # Summary statistics (committed)
│       ├── metadata.json        # Environment and configuration data (committed)
│       ├── results.csv          # CSV export of results (committed)
│       ├── statistical_analysis_report.json (committed)
│       └── performance_profiling_report.json (committed)
│
├── database/
│   ├── results.db               # SQLite database (committed)
│   ├── schema.sql               # Database schema definition
│   └── schema_enhancement.sql   # Schema updates and enhancements
│
├── requirements/
│   ├── base.txt                 # Core dependencies
│   ├── python.txt               # Python solver dependencies
│   └── export.txt               # Export functionality dependencies
│
├── logs/                        # Log files (local only, .gitignore)
│   └── benchmark.log            # Structured execution logs
│
└── tests/                       # Test suite (cleaned up)
    ├── unit/                    # Unit tests
    ├── integration/             # Integration tests
    └── fixtures/                # Test data and fixtures
```

---

## Core Components (Simplified)

### 1. GitHub Actions Workflows (Minimal)

#### deploy-pages.yml - Static File Publishing Only
```yaml
# Key features:
# - Triggers on push to main branch
# - Publishes PRE-BUILT docs/ folder to GitHub Pages
# - No benchmark execution in CI
# - Fast and reliable deployment
# - Preserves PR preview subdirectories
```

#### pr-preview.yml - Pull Request Preview System
```yaml
# Key features:
# - Auto-deploy PR previews to gh-pages/pr-preview/pr-{number}/
# - Lightweight benchmark with scipy,clarabel_cvxpy,scs_cvxpy and light_set
# - Auto-cleanup when PR closed
# - Preview banners and metadata injection
# - Comments with preview URLs on PRs
```

### 2. Configuration Management (Simplified)

#### benchmark_config.yaml - Local Execution Settings
```yaml
benchmark:
  timeout: 300                   # Solver timeout in seconds
  parallel_jobs: 1               # CPU core utilization (fair comparison)
  problem_sets:
    light_set: "problems/light_set"   # Only local problems
  
reporting:
  formats: ["html", "json", "csv"]
  include_environment_info: true
  include_statistical_analysis: true

database:
  path: "database/results.db"   # Local SQLite database

output:
  reports_dir: "docs"            # Pre-built HTML files
```

#### solvers.yaml - Python Solver Definitions Only
```yaml
solvers:
  scipy:
    type: python
    module: scripts.solvers.python.scipy_runner
    class: ScipySolver
    
  cvxpy:
    type: python  
    module: scripts.solvers.python.cvxpy_runner
    class: CVXPYSolver
    backends: [CLARABEL, SCS, ECOS, OSQP]  # Multi-backend support
```

### 3. Benchmark Execution Engine

#### runner.py - Orchestration Controller
```python
class BenchmarkRunner:
    """
    Main benchmark execution controller
    - Manages parallel execution across problems and solvers
    - Handles timeouts and error recovery
    - Collects and aggregates results
    - Provides progress reporting and logging
    """
```

#### solver_interface.py - Abstraction Layer
```python
class SolverInterface(ABC):
    """
    Abstract base class for all solvers
    - Standardized solve() method signature
    - Consistent result format (SolverResult dataclass)
    - Error handling and timeout management
    - Environment compatibility checking
    """
```

#### backend_selector.py - CVXPY Backend Management
```python
class BackendSelector:
    """
    Intelligent CVXPY backend selection
    - Problem type compatibility matrix
    - Solver capability detection
    - Fallback mechanism for failed backends
    - Performance-based recommendations
    """
```

### 4. Database Layer

#### Schema Design
```sql
-- Core tables for benchmark results
CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    environment_info TEXT,
    git_commit TEXT,
    configuration_hash TEXT
);

CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    benchmark_id INTEGER,
    solver_name TEXT,
    problem_name TEXT,
    problem_type TEXT,
    solve_time REAL,
    status TEXT,
    objective_value REAL,
    duality_gap REAL,
    error_message TEXT,
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks (id)
);

CREATE TABLE problems (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    problem_type TEXT,
    file_path TEXT,
    variables INTEGER,
    constraints INTEGER,
    difficulty TEXT
);

CREATE TABLE solvers (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    solver_type TEXT,
    version TEXT,
    backend TEXT,
    environment TEXT
);
```

#### models.py - Data Models
```python
# SQLAlchemy models with validation
class BenchmarkResult:
    """Complete benchmark result with environment context"""
    
class SolverResult:
    """Individual solver execution result"""
    
class ProblemMetadata:
    """Problem characteristics and classification"""
```

### 5. Reporting System

#### simple_html_generator.py - Report Generation
```python
class SimpleHTMLGenerator:
    """
    Generates comprehensive HTML reports
    - Bootstrap 5 styling with responsive design
    - Chart.js visualizations for performance data
    - Multiple report types (dashboard, comparison, analysis)
    - Author attribution and metadata integration
    """
```

#### data_publisher.py - Data Export
```python
class DataPublisher:
    """
    Multi-format data publishing
    - JSON export for programmatic access
    - CSV export for spreadsheet analysis
    - Metadata export for reproducibility
    - Validation and quality assurance
    """
```

---

## Data Formats and Standards

### Result Data Schema
```json
{
  "benchmark_metadata": {
    "timestamp": "2025-12-15T10:30:00Z",
    "environment": {
      "platform": "macOS-14.5-arm64-arm-64bit",
      "python_version": "3.12.2",
      "cpu_cores": 8,
      "memory_gb": 24.0,
      "timezone": "JST (UTC+9.0)"
    },
    "configuration": {
      "timeout": 300,
      "parallel_jobs": 1,
      "problem_set": "light_set"
    }
  },
  "results": [
    {
      "solver": "CLARABEL (via CVXPY)",
      "problem": "SIMPLE_LP",
      "problem_type": "LP", 
      "solve_time": 0.0025,
      "status": "optimal",
      "objective_value": 5.98e-10,
      "duality_gap": null,
      "variables": 2,
      "constraints": 2
    }
  ]
}
```

### Problem Classification
```python
class ProblemType(Enum):
    LP = "Linear Programming"
    QP = "Quadratic Programming" 
    SOCP = "Second-Order Cone Programming"
    SDP = "Semidefinite Programming"
    
class DifficultyLevel(Enum):
    TRIVIAL = "Small problems for testing"
    EASY = "Standard benchmark problems"
    MEDIUM = "Challenging real-world problems"
    HARD = "Large-scale or ill-conditioned problems"
```

---

## Performance Optimization

### Parallel Execution Strategy
```python
# Problem-level parallelization with CPU fairness
parallel_jobs: 1  # Fair CPU utilization for consistent benchmarking

# Solver-level sequential execution
# Ensures each solver gets dedicated CPU resources
# Prevents resource contention affecting results
```

### Caching and Storage
```python
# Problem file caching for external storage
# Solver installation state caching
# Result incremental updates
# Compressed data transfer for GitHub Pages
```

### GitHub Actions Optimization
```yaml
# Artifact handling for database persistence
# Staged deployment with dependency caching
# Efficient matrix execution for multiple environments
# Resource-aware job scheduling
```

---

## Extension Points

### Adding New Solvers
```python
# 1. Implement SolverInterface
class NewSolver(SolverInterface):
    def solve(self, problem: Problem) -> SolverResult:
        # Implementation here
        pass

# 2. Add configuration
# config/solvers.yaml:
new_solver:
  type: python
  module: scripts.solvers.python.new_solver
  class: NewSolver

# 3. Add dependencies
# requirements/python.txt:
new-solver-package>=1.0.0
```

### Adding Problem Types
```python
# 1. Extend problem loader
class ProblemLoader:
    def load_new_type(self, file_path: str) -> Problem:
        # Implementation for new format
        pass

# 2. Update classification
class ProblemClassifier:
    def classify_new_type(self, problem: Problem) -> ProblemMetadata:
        # Classification logic
        pass
```

---

## Security and Validation

### Input Validation
```python
# Problem file format validation
# Configuration value range checking  
# Solver result consistency verification
# Database input sanitization
```

### Environment Isolation
```python
# GitHub Actions sandboxed execution
# Explicit dependency management
# Timeout enforcement for solver execution
# Resource limit enforcement
```

### Data Integrity
```python
# Result validation against expected ranges
# Cross-solver consistency checks
# Environment reproducibility verification
# Configuration hash validation
```

---

## Monitoring and Operations

### Logging Framework
```python
# Structured logging with levels (DEBUG, INFO, WARNING, ERROR)
# Execution time measurement and reporting
# Error context capture for debugging
# Performance metrics collection
```

### Error Handling
```python
# Graceful solver failure handling
# Partial result collection on timeout
# Automatic retry for transient failures
# Comprehensive error reporting
```

### Deployment Pipeline
```python
# GitHub Actions automated testing
# Staged deployment with validation
# Rollback capability for failed deployments
# Environment consistency verification
```

---

*This detailed design document provides comprehensive technical specifications for implementation. For high-level concepts and project vision, see [basic_design.md](basic_design.md).*

---

## Current Implementation Status

### ✅ Completed Components (Sprint 1-2)
- **Core Simplification**: Removed Octave support, external storage components
- **GitHub Actions**: Simplified to static file publishing only
- **Local Workflow**: Complete benchmark execution and HTML report generation
- **Test Organization**: Cleaned up and organized test files in proper directory structure
- **Data Cleanup**: Removed test data, verified legitimate results only

### 🔄 Current Capabilities
- **Solvers**: 5 Python-based solvers (SciPy, CLARABEL, SCS, ECOS, OSQP)
- **Problems**: 8 problems from light_set (LP, QP, SOCP, SDP)
- **Workflow**: `python main.py --all` runs complete local benchmark + report generation
- **Reports**: 7 HTML files + 6 data files generated in docs/
- **Database**: SQLite with 53 legitimate benchmark results

### 📋 Implementation Notes
- **Local-First Approach**: All benchmark execution happens locally, CI only publishes
- **Pre-Built Artifacts**: HTML reports and data files are committed to repository
- **Clean Architecture**: Removed complexity, focused on reliability
- **Test Coverage**: Organized test suite with unit, integration, and debug tests

*Last Updated: June 2025 (Sprint 2 Complete)*

---

## 🚀 Next Major Enhancement: Public Reporting System (Sprint 5+)

### Overview
Transform the benchmark system to support meaningful public reporting with real-world problem instances from established optimization libraries.

### Key Features
1. **Standard Benchmark Libraries**: Integration with DIMACS and SDPLIB problem sets
2. **Solver Version Tracking**: Record and display backend versions for reproducibility
3. **Aggregated Reporting**: Statistical analysis across multiple benchmark runs
4. **Enhanced Documentation**: Comprehensive guides for problem library integration

---

## Enhanced Architecture Design

### 1. Problem Library Integration System

#### Extended Problem Registry Architecture
```yaml
# Enhanced problem_registry.yaml structure
problem_libraries:
  light_set:
    source: "internal"
    description: "Internal test problems for basic validation"
    problems: [existing structure]
    
  DIMACS:
    source: "external"
    library_url: "https://github.com/vsdp/DIMACS.git"
    file_format: "mat"
    description: "DIMACS optimization problems from established library"
    problems:
      - name: "problem_name"
        library: "DIMACS"
        file_path: "problems/DIMACS/category/problem.mat"
        file_type: "sedumi_mat"
        problem_class: "SDP"
        metadata:
          variables: 100
          constraints: 50
          source_category: "control"
          
  SDPLIB:
    source: "external" 
    library_url: "https://github.com/vsdp/SDPLIB.git"
    file_format: "dat-s"
    description: "SDPLIB semidefinite programming test problems"
    problems:
      - name: "problem_name"
        library: "SDPLIB"
        file_path: "problems/SDPLIB/category/problem.dat-s"
        file_type: "sdpa_sparse"
        problem_class: "SDP"
        metadata:
          variables: 200
          constraints: 100
          source_category: "graph"
```

#### Problem Loader Architecture Enhancement
```python
# Enhanced ProblemData class
class ProblemData:
    def __init__(self, name: str, problem_class: str, library: str = "internal",
                 file_type: str = "python", source_metadata: dict = None, ...):
        # Existing fields
        self.name = name
        self.problem_class = problem_class
        
        # New library tracking fields
        self.library = library  # "light_set", "DIMACS", "SDPLIB"
        self.file_type = file_type  # "python", "mps", "qps", "sedumi_mat", "sdpa_sparse"
        self.source_metadata = source_metadata or {}
        
        # Enhanced metadata for reporting
        self.library_version = None  # Library version when downloaded
        self.library_url = None  # Source URL for external libraries
        
# Enhanced problem loader with library support
class EnhancedProblemLoader:
    def __init__(self):
        self.format_loaders = {
            'python': self.load_python_problem,
            'mps': self.load_mps_file,
            'qps': self.load_qps_file,
            'sedumi_mat': self.load_sedumi_mat,  # DIMACS .mat files
            'sdpa_sparse': self.load_sdpa_sparse  # SDPLIB .dat-s files
        }
    
    def load_problem_by_library(self, library: str, problem_name: str) -> ProblemData:
        """Load problem with library context"""
        
    def get_problems_by_library(self, library: str) -> List[ProblemData]:
        """Get all problems from specific library"""
```

#### Directory Structure for External Libraries
```
problems/
├── light_set/                 # Existing internal problems
│   ├── lp/, qp/, socp/, sdp/
│   └── ...
├── DIMACS/                    # DIMACS problem library
│   ├── control/               # Organized by problem category
│   │   ├── problem1.mat
│   │   └── problem2.mat
│   ├── graph/
│   │   └── ...
│   └── LICENSE                # Library license
├── SDPLIB/                    # SDPLIB problem library  
│   ├── graph/
│   │   ├── problem1.dat-s
│   │   └── problem2.dat-s
│   ├── combinatorial/
│   │   └── ...
│   └── LICENSE
└── problem_registry.yaml     # Enhanced registry with library metadata
```

### 2. Solver Version Tracking System

#### Enhanced Database Schema
```sql
-- Enhanced solvers table with version as primary component
CREATE TABLE IF NOT EXISTS solvers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    backend TEXT,                    -- CLARABEL, SCS, ECOS, etc.
    environment TEXT,                -- python, system package
    metadata TEXT,                   -- JSON: installation method, dependencies
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version, backend)   -- Treat different versions as distinct solvers
);

-- Enhanced results table with solver version tracking
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id INTEGER NOT NULL,
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,    -- New: track solver version
    solver_backend TEXT,             -- New: track specific backend (for CVXPY)
    problem_name TEXT NOT NULL,
    problem_library TEXT DEFAULT 'light_set',  -- New: track problem source
    solve_time REAL,
    status TEXT,
    objective_value REAL,
    duality_gap REAL,
    iterations INTEGER,
    error_message TEXT,
    solver_info TEXT,
    run_id TEXT,                     -- New: track multiple runs for aggregation
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks (id)
);

-- New table for aggregated results view
CREATE TABLE IF NOT EXISTS aggregated_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    problem_name TEXT NOT NULL,
    problem_library TEXT NOT NULL,
    
    -- Aggregated statistics (median of recent 5 runs)
    median_solve_time REAL,
    median_objective_value REAL,
    success_rate REAL,              -- Percentage of successful runs
    
    -- Run metadata
    runs_count INTEGER,             -- Number of runs included in aggregation
    most_recent_run DATETIME,
    aggregation_window INTEGER DEFAULT 5,
    
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(solver_name, solver_version, problem_name, problem_library)
);
```

#### Solver Version Detection Architecture
```python
# Enhanced solver interface with version tracking
class SolverInterface(ABC):
    def __init__(self, name: str, timeout: float = 300.0):
        self.name = name
        self.timeout = timeout
        self.version = self.detect_version()      # Auto-detect version
        self.backend = self.detect_backend()      # Detect specific backend
        self.metadata = self.gather_metadata()   # Installation details
    
    @abstractmethod
    def detect_version(self) -> str:
        """Detect and return solver version"""
        pass
    
    def detect_backend(self) -> Optional[str]:
        """Detect specific solver backend (for CVXPY solvers)"""
        return None
        
    def gather_metadata(self) -> dict:
        """Gather installation and environment metadata"""
        return {
            'detection_method': 'auto',
            'detected_at': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': platform.platform()
        }

# Enhanced CVXPY solver with version tracking
class CvxpySolver(SolverInterface):
    def __init__(self, name: str, backend: str, timeout: float = 300.0, **kwargs):
        self.backend_name = backend
        super().__init__(name, timeout)
    
    def detect_version(self) -> str:
        """Detect CVXPY and backend versions"""
        import cvxpy as cp
        cvxpy_version = cp.__version__
        
        # Detect backend-specific version
        backend_version = self._detect_backend_version(self.backend_name)
        
        return f"cvxpy-{cvxpy_version}+{self.backend_name}-{backend_version}"
    
    def detect_backend(self) -> str:
        return self.backend_name
        
    def _detect_backend_version(self, backend: str) -> str:
        """Detect specific backend version"""
        try:
            if backend == "CLARABEL":
                import clarabel
                return clarabel.__version__
            elif backend == "SCS":
                import scs
                return scs.__version__
            elif backend == "ECOS":
                import ecos
                return ecos.__version__
            elif backend == "OSQP":
                import osqp
                return osqp.__version__
            else:
                return "unknown"
        except ImportError:
            return "not-installed"
```

### 3. Aggregated Reporting System

#### Multi-Run Execution Architecture
```python
class AggregatedBenchmarkRunner:
    """Enhanced benchmark runner with multi-run support"""
    
    def __init__(self, config_path: str, aggregation_window: int = 5):
        self.aggregation_window = aggregation_window
        self.run_id = self.generate_run_id()
        
    def run_benchmark_with_aggregation(self, problems: List[str], 
                                     solvers: List[str]) -> dict:
        """Run benchmark and update aggregated results"""
        
        # Execute single benchmark run
        results = self.run_single_benchmark(problems, solvers)
        
        # Update aggregated results
        self.update_aggregated_results(results)
        
        return self.get_aggregated_results()
    
    def update_aggregated_results(self, new_results: List[dict]):
        """Update aggregated results with new run data"""
        for result in new_results:
            # Get recent runs for this solver+problem combination
            recent_runs = self.get_recent_runs(
                result['solver_name'], 
                result['solver_version'],
                result['problem_name'],
                result['problem_library'],
                limit=self.aggregation_window
            )
            
            # Calculate aggregated statistics
            aggregated = self.calculate_aggregated_stats(recent_runs)
            
            # Store/update aggregated result
            self.store_aggregated_result(aggregated)
```

#### Enhanced HTML Report Generator
```python
class EnhancedHTMLGenerator:
    """Enhanced HTML generator with library and version support"""
    
    def generate_solver_comparison_report(self) -> str:
        """Generate solver comparison with version information"""
        
        # Group results by solver name+version
        solver_results = self.get_aggregated_results_by_solver()
        
        # Generate comparison table with version columns
        html = self.render_solver_version_table(solver_results)
        
        return html
    
    def generate_library_breakdown_report(self) -> str:
        """Generate report showing results by problem library"""
        
        library_results = self.get_results_by_library()
        
        # Separate sections for each library
        html_sections = []
        for library, results in library_results.items():
            section = self.render_library_section(library, results)
            html_sections.append(section)
            
        return self.combine_html_sections(html_sections)
    
    def render_solver_version_table(self, solver_results: dict) -> str:
        """Render table showing different solver versions"""
        # Table headers: Solver | Version | Backend | Problem Success Rate | Median Time
```

### 4. Implementation Architecture Overview

#### Component Interaction Flow
```
ENHANCED BENCHMARK FLOW:

1. Problem Loading (Enhanced)
   ├── Load problem_registry.yaml with library metadata
   ├── Identify problem library (light_set, DIMACS, SDPLIB)
   ├── Use appropriate format loader (python, mps, qps, mat, dat-s)
   └── Create ProblemData with library context

2. Solver Initialization (Enhanced)
   ├── Auto-detect solver versions and backends
   ├── Store solver metadata in database
   ├── Create solver instances with version tracking
   └── Validate solver capabilities for problem types

3. Benchmark Execution (Enhanced)
   ├── Generate unique run_id for this execution
   ├── Execute problems across solver+version combinations
   ├── Store individual results with run_id and version info
   └── Update aggregated results after completion

4. Report Generation (Enhanced)
   ├── Generate solver comparison with version breakdown
   ├── Generate library-specific performance reports
   ├── Show aggregated statistics (median of recent 5 runs)
   ├── Include problem library attribution
   └── Export enhanced data formats

5. Database Management (Enhanced)
   ├── Track multiple runs per solver+problem combination  
   ├── Maintain aggregated results table
   ├── Store solver version and library metadata
   └── Support historical analysis and version comparison
```

#### File Organization Updates
```
scripts/
├── benchmark/
│   ├── runner.py                    # Enhanced with multi-run support
│   ├── aggregated_runner.py         # New: aggregation-aware runner
│   ├── problem_loader.py            # Enhanced with library support
│   ├── solver_interface.py          # Enhanced with version detection
│   └── library_manager.py           # New: manage external libraries
│
├── solvers/
│   └── python/
│       ├── cvxpy_runner.py          # Enhanced with version detection
│       ├── scipy_runner.py          # Enhanced with version detection
│       └── version_detector.py      # New: version detection utilities
│
├── database/
│   ├── models.py                    # Enhanced with new schema
│   ├── aggregation.py               # New: aggregated results management
│   └── migration.py                 # New: schema migration utilities
│
├── reporting/
│   ├── simple_html_generator.py     # Enhanced with library/version support
│   ├── library_report_generator.py  # New: library-specific reports
│   └── aggregation_reporter.py      # New: aggregated statistics reporting
│
├── utils/
│   ├── library_downloader.py        # New: download and manage external libraries
│   ├── format_converters.py         # New: problem format conversion utilities
│   └── version_utils.py             # New: version detection and comparison
│
└── external/
    ├── dimacs_loader.py              # New: DIMACS .mat file loader
    ├── sdplib_loader.py              # New: SDPLIB .dat-s file loader
    └── library_validator.py          # New: external library validation
```

---

### Implementation Priority and Dependencies

#### Phase 1: Foundation (Weeks 1-2)
1. **Database Schema Enhancement** (High Priority)
   - Add solver version tracking to schema
   - Add problem library tracking
   - Add aggregated results table
   - Create migration scripts

2. **Solver Version Detection** (High Priority)
   - Implement version detection for all existing solvers
   - Update solver interfaces with version tracking
   - Test version detection across different environments

#### Phase 2: External Library Integration (Weeks 3-4)
3. **Problem Library Management** (Medium Priority)
   - Create library download and organization system
   - Implement DIMACS .mat file loader
   - Implement SDPLIB .dat-s file loader
   - Extend problem registry with library metadata

4. **Enhanced Problem Loading** (Medium Priority)
   - Update problem loader with library support
   - Test loading problems from all three libraries
   - Validate problem data consistency

#### Phase 3: Aggregation and Reporting (Weeks 5-6)
5. **Multi-Run Aggregation** (Medium Priority)
   - Implement run tracking and aggregation logic
   - Create aggregated results calculation
   - Test aggregation with multiple benchmark runs

6. **Enhanced Reporting** (Low Priority)
   - Update HTML reports with version and library information
   - Create library-specific report sections
   - Add aggregated statistics display

#### Phase 4: Documentation and Polish (Week 7)
7. **Documentation Updates** (Low Priority)
   - Update README with new features
   - Create library integration guide
   - Document version tracking and aggregation

### Risk Analysis and Mitigation

#### High Risk Areas:
1. **External Library Integration**: File format compatibility issues
2. **Version Detection**: Inconsistent version reporting across solvers  
3. **Database Migration**: Schema changes affecting existing data
4. **Performance Impact**: Loading large external problem sets

#### Mitigation Strategies:
1. **Incremental Testing**: Test each component independently
2. **Backward Compatibility**: Maintain support for existing light_set problems
3. **Fallback Mechanisms**: Graceful handling of version detection failures
4. **Performance Monitoring**: Track impact of external libraries on execution time