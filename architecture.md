# Optimization Solver Benchmark System Architecture

## System Overview

A GitHub Actions-based benchmark system for optimization solvers (SDP, SOCP, LP, QP). Automatically executes solvers in Python/Octave environments, stores results in SQLite, and publishes them via GitHub Pages.

## Overall Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub Repo   │    │  GitHub Actions  │    │  GitHub Pages   │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │Config Files │ │───▶│ │Benchmark     │ │───▶│ │Static Site  │ │
│ │Problems     │ │    │ │Execution     │ │    │ │Dashboard    │ │
│ │Scripts      │ │    │ │Environment   │ │    │ │Reports      │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   SQLite DB      │
                       │   (results.db)   │
                       └──────────────────┘
```

## File and Folder Structure

```
optimizer-benchmark/
├── README.md
├── architecture.md
├── LICENSE
│
├── .github/
│   └── workflows/
│       ├── benchmark.yml          # メインベンチマーク実行
│       ├── deploy.yml             # GitHub Pages デプロイ
│       └── test.yml               # テスト実行
│
├── config/
│   ├── benchmark_config.yaml     # ベンチマーク設定
│   ├── solvers.yaml              # ソルバー定義
│   └── environments.yaml         # 環境設定
│
├── problems/
│   ├── light_set/                # Phase 1: Lightweight problems
│   │   ├── lp/                   # Linear programming problems
│   │   │   ├── afiro.mps
│   │   │   ├── adlittle.mps
│   │   │   └── blend.mps
│   │   ├── qp/                   # Quadratic programming problems
│   │   │   ├── qp_example1.qps
│   │   │   └── qp_example2.qps
│   │   ├── socp/                 # Second-order cone programming
│   │   │   └── portfolio_optimization.py
│   │   └── sdp/                  # Semidefinite programming problems
│   │       ├── arch0.dat-s
│   │       └── control1.dat-s
│   ├── medium_set/               # Phase 2: External medium problems
│   │   └── external_urls.yaml    # External storage URLs
│   ├── large_set/                # Phase 2: External large problems  
│   │   └── external_urls.yaml    # External storage URLs
│   └── problem_registry.yaml     # Problem metadata
│
├── scripts/
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── runner.py             # Benchmark execution engine
│   │   ├── solver_interface.py   # Solver abstraction layer
│   │   ├── problem_loader.py     # Problem loading
│   │   ├── result_collector.py   # Result collection
│   │   └── environment_info.py   # Environment information gathering
│   │
│   ├── solvers/
│   │   ├── python/
│   │   │   ├── __init__.py
│   │   │   ├── cvxpy_runner.py   # CVXPY execution with multiple backends
│   │   │   └── scipy_runner.py   # SciPy execution
│   │   └── octave/
│   │       ├── __init__.py
│   │       ├── octave_runner.py  # Octave execution control
│   │       ├── linprog_solver.m  # Linear programming
│   │       ├── quadprog_solver.m # Quadratic programming
│   │       └── problem_loader.m  # Problem loading utilities
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql            # Database schema
│   │   ├── models.py             # Data models
│   │   ├── migrations.py         # Migrations
│   │   └── queries.py            # Query definitions
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── data_publisher.py     # JSON/CSV data publishing
│   │   ├── html_generator.py     # Simple HTML generation
│   │   ├── statistics.py         # Basic statistical calculations
│   │   └── export.py             # Multi-format data export
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py      # Configuration loading
│       ├── file_handler.py       # File operations
│       ├── logger.py             # Log management
│       └── validation.py         # Data validation
│
├── templates/
│   ├── base.html                 # Base template
│   ├── dashboard.html            # Dashboard
│   ├── solver_comparison.html    # Solver comparison
│   ├── problem_analysis.html     # Problem-wise analysis
│   └── environment_info.html     # Environment information display
│
├── static/
│   ├── css/
│   │   ├── main.css
│   │   └── charts.css
│   ├── js/
│   │   ├── main.js
│   │   ├── charts.js
│   │   └── interactive.js
│   └── images/
│       └── logo.png
│
├── docs/                         # GitHub Pages output
│   ├── index.html               # Auto-generated
│   ├── data/
│   │   ├── results.json         # Result data
│   │   └── summary.json         # Summary data
│   └── assets/
│       ├── css/
│       ├── js/
│       └── images/
│
├── database/
│   ├── results.db               # SQLite database
│   └── backups/
│       └── results_backup.db
│
├── logs/
│   ├── benchmark.log
│   ├── error.log
│   └── debug.log
│
├── tests/
│   ├── __init__.py
│   ├── test_runner.py
│   ├── test_solvers.py
│   ├── test_database.py
│   ├── test_reporting.py
│   └── fixtures/
│       ├── sample_problems/
│       └── expected_results/
│
└── requirements/
    ├── base.txt                 # Base dependencies
    ├── python.txt               # Python solvers
    ├── octave.txt               # Octave package list
    └── dev.txt                  # Development dependencies
```

## Key Components

### 1. GitHub Actions Workflows

#### benchmark.yml
```yaml
# Scheduled execution (daily at UTC 2:00)
# Manual trigger support
# Multi-environment matrix execution
```

#### deploy.yml  
```yaml
# Auto-execution after benchmark completion
# Static site generation
# GitHub Pages deployment
```

### 2. Configuration Management System

#### config/benchmark_config.yaml
```yaml
# Benchmark execution settings
# Timeout configuration
# Parallel execution count
# Output format
```

#### config/solvers.yaml
```yaml
# Solver definitions
# Installation procedures
# Execution parameters
# Prerequisites
```

### 3. Benchmark Execution Engine

#### scripts/benchmark/runner.py
- Overall benchmark execution control
- Parallel execution management
- Error handling
- Progress reporting

#### scripts/benchmark/solver_interface.py
- Solver abstraction layer
- Unified interface
- Result standardization
- Error processing

### 4. Database Layer

#### database/schema.sql
```sql
-- Execution results table
-- Environment information table
-- Solver information table
-- Problem information table
```

#### database/models.py
- SQLAlchemy model definitions
- Data validation
- Relationship definitions

### 5. Reporting System

#### reporting/html_generator.py
- Dynamic HTML generation
- Template processing
- Static asset management

#### reporting/chart_generator.py
- Performance graphs
- Comparison charts
- Statistical visualization

## Data Flow

```
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
```

## Execution Flow

### 1. Benchmark Execution Phase
```python
1. Load configuration files
2. Gather environment information
3. Load and validate problem files
4. Set up solver environments
5. Execute benchmarks (parallel)
6. Collect and validate results
7. Store in database
```

### 2. Report Generation Phase
```python
1. Retrieve data from database
2. Calculate statistics
3. Generate charts
4. Generate HTML
5. Output static files
6. Deploy to GitHub Pages
```

## Extension Points

### Phase 2 Extensions (Current)

#### Octave Support
```
├── scripts/solvers/octave/
│   ├── octave_runner.py
│   ├── linprog_solver.m
│   ├── quadprog_solver.m
│   └── problem_loader.m
├── config/octave_config.yaml
└── .github/workflows/octave_test.yml
```

#### External Storage (Simplified)
```
├── scripts/storage/
│   ├── external_storage.py
│   └── external_manager.py
├── problems/medium_set/external_urls.yaml
└── problems/large_set/external_urls.yaml
```

#### Data Publishing Infrastructure
```
├── scripts/reporting/data_publisher.py
├── docs/data/
│   ├── results.json
│   ├── summary.json
│   └── metadata.json
└── scripts/api/ (basic endpoints)
```

### Future Extensions (Optional)

#### Advanced Analytics (Optional)
```
├── scripts/analytics/ (moved from core)
├── advanced_reports/
└── machine_learning_models/
```

#### MATLAB Support (Future)
```
├── scripts/solvers/matlab/
├── config/matlab_config.yaml
└── api/result_submission.py
```

## Security Considerations

### 1. Secret Management
- API key management via GitHub Secrets
- Commercial solver license information protection
- External configuration via environment variables

### 2. Execution Environment Isolation
- Docker container usage consideration
- Explicit dependency management
- Sandbox execution

### 3. Input Validation
- Problem file validation
- Configuration value range checking
- SQL injection prevention

## Performance Optimization

### 1. Parallel Execution
- Problem-level parallelization
- Solver-level parallelization
- GitHub Actions matrix utilization

### 2. Caching Strategy
- Problem file caching
- Solver installation caching
- Dependency caching

### 3. Result Efficiency
- Incremental updates
- Differential reporting
- Compressed data transfer

## Monitoring and Operations

### 1. Log Management
- Structured logging
- Level-based log output
- Execution time measurement

### 2. Error Handling
- Staged error processing
- Partial failure tolerance
- Automatic retry functionality

### 3. Notification System
- Automatic GitHub Issues creation
- Execution completion notifications
- Error notifications

This architecture enables the construction of a highly scalable benchmark system that supports staged development and operations.