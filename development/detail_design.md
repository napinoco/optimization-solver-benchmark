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