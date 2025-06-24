# Optimization Solver Benchmark System

A comprehensive open-source platform for benchmarking optimization solvers across multiple problem types (LP, QP, SOCP, SDP). Automated execution via GitHub Actions with interactive HTML reports published to GitHub Pages.

**Project Vision**: Create the leading open-source platform for benchmarking optimization solvers across multiple problem types, providing researchers and practitioners with comprehensive performance insights through automated execution, transparent reporting, and fair baseline comparisons.

**Core Mission**: "Regularly benchmark publicly available solvers and publish the results as data"

## External Problem Libraries

This benchmark system uses the following external problem libraries:

### DIMACS Library
- **Repository**: [https://github.com/vsdp/DIMACS.git](https://github.com/vsdp/DIMACS.git)
- **Current Commit**: `a1a14d1` - FILTER/filtinf1.mat.gz: Fix problem vector 'c'.
- **Problems**: 47 optimization challenge problems in SeDuMi .mat format

### SDPLIB Library  
- **Repository**: [https://github.com/vsdp/SDPLIB.git](https://github.com/vsdp/SDPLIB.git)
- **Current Commit**: `2e568e1` - README.md: Remove dead link.
- **Problems**: 92+ semidefinite programming problems in SDPA .dat-s format

Both libraries are included as Git submodules and automatically loaded by the benchmark system.

## 🎯 Current Status: Production Ready

- ✅ **Complete External Library Integration**: 139+ problems (DIMACS + SDPLIB)
- ✅ **Comprehensive Solver Support**: 9 solvers (SciPy, CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HiGHS via CVXPY)
- ✅ **Unified ProblemData Architecture**: SeDuMi-like format for all problem types
- ✅ **Professional Reporting**: Interactive HTML dashboards with problem structure analysis
- ✅ **GitHub Actions CI/CD**: Automated benchmarking and report publishing

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git and GitHub account
- Virtual environment recommended

### Installation
```bash
git clone --recursive https://github.com/napinoco/optimization-solver-benchmark.git
cd optimization-solver-benchmark
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements/base.txt
pip install -r requirements/python.txt
```

### Run Benchmarks
```bash
# Validate environment and external libraries
python main.py --validate

# Complete benchmark with all libraries
python main.py --all

# Benchmark specific libraries
python main.py --benchmark --problem-set dimacs
python main.py --benchmark --problem-set sdplib

# Generate reports only  
python main.py --report

# Dry-run mode for testing
python main.py --benchmark --problems nb --dry-run
```

## 🎯 Design Philosophy: Fair Baseline Benchmarking

This system prioritizes **unbiased solver comparison** through minimal configuration:

- **Solver Defaults**: Uses each solver's default parameters to avoid optimization bias
- **Baseline Performance**: Establishes genuine "out of the box" capabilities
- **Transparent Comparison**: Prevents inadvertent parameter tuning favoring specific solvers
- **Reproducible Results**: Standardized environments ensure consistent benchmarking
- **Open Data**: Results published as accessible JSON/CSV for research use

## 📊 Current System Capabilities

### Supported Solvers ✅
- **SciPy**: Linear programming solver
- **CVXPY Multi-Backend**: 
  - CLARABEL (modern Rust-based solver)
  - SCS (Splitting Conic Solver)
  - ECOS (Embedded Conic Solver)
  - OSQP (Operator Splitting QP)
  - CVXOPT (Convex optimization)
  - SDPA (Semidefinite programming)
  - SCIP (Mixed-integer programming)
  - HiGHS (Linear/mixed-integer programming)

### Problem Type Coverage ✅
- **LP**: Linear Programming (12+ results, 100% success rate)
- **QP**: Quadratic Programming (6+ results, 100% success rate)
- **SOCP**: Second-Order Cone Programming (31+ results, ~43% success rate)
- **SDP**: Semidefinite Programming (38+ results, ~29% success rate)

### Key Features ✅
- **External Library Integration**: DIMACS (47 problems) + SDPLIB (92+ problems)
- **Unified Problem Format**: SeDuMi-like architecture for all problem types
- **Comprehensive Version Tracking**: Solver backend versions and Git commit recording
- **Interactive Reports**: Bootstrap 5 dashboards with Chart.js visualizations
- **Problem Structure Analysis**: Automatic variable/constraint counting and classification
- **GitHub Pages Deployment**: Automatic report publishing with data exports
- **Validation Framework**: Data quality checks and error detection
- **Dry-Run Mode**: Testing without database pollution

## 📁 Project Structure

```
optimization-solver-benchmark/
├── 📋 Documentation
│   ├── README.md                    # This file
│   ├── CLAUDE.md                   # AI development context
│   └── docs/
│       ├── guides/                 # Setup and usage guides
│       └── development/            # Development documentation
│
├── ⚙️ Configuration
│   └── config/
│       ├── site_config.yaml        # Site and overview settings
│       ├── solver_registry.yaml    # Solver display names
│       └── problem_registry.yaml   # Problem metadata
│
├── 🧪 Core System
│   ├── main.py                     # Main entry point
│   └── scripts/
│       ├── benchmark/              # Benchmark execution engine
│       ├── solvers/python/         # Solver implementations
│       ├── data_loaders/           # Problem loading (MAT/DAT)
│       │   ├── problem_loader.py   # Unified ProblemData class
│       │   └── python/             # Format-specific loaders
│       ├── external/               # External library loaders
│       ├── utils/                  # Problem structure analysis
│       ├── database/               # SQLite data models
│       └── reporting/              # HTML report generation
│
├── 📊 Problems & Results
│   ├── problems/
│   │   ├── DIMACS/                 # External DIMACS library (47 problems)
│   │   ├── SDPLIB/                 # External SDPLIB library (92+ problems)
│   │   └── light_set/              # Internal synthetic problems
│   ├── database/                   # SQLite results with version tracking
│   └── docs/                       # Generated reports and data exports
│
├── 🔧 Dependencies
│   └── requirements/
│       ├── base.txt                # Core dependencies
│       └── python.txt              # Python solver dependencies
│
└── 🚀 CI/CD
    └── .github/
        └── workflows/              # GitHub Actions and documentation
```

## 📈 Performance Results

### Current Success Rates
```
Problem Type | Total Results | Success Rate | Top Performers
LP           | 12 results   | 100%        | SciPy + CLARABEL + SCS + ECOS + OSQP
QP           | 6 results    | 100%        | SciPy + CLARABEL + SCS + ECOS + OSQP  
SOCP         | 31 results   | ~43%        | CLARABEL + SCS + ECOS
SDP          | 38 results   | ~29%        | CLARABEL + SCS
```

### External Library Performance
```
Library  | Problems | Format        | Success Rate | Notes
DIMACS   | 47       | SeDuMi .mat   | Variable     | Mixed problem types
SDPLIB   | 92+      | SDPA .dat-s   | ~30%         | Challenging SDP problems
Total    | 139+     | Mixed         | ~37%         | System average
```

## 🔧 GitHub Actions Integration

### Automatic Execution
- **Push to main**: Triggers comprehensive benchmarking
- **Pull requests**: Validation and testing
- **Manual triggers**: Custom solver/problem selection

### Manual Trigger Options
Access via GitHub Actions tab → "Run workflow":
- **Problem Set**: Choose from external, dimacs, sdplib, or light_set
- **Solvers**: Select specific solvers to benchmark
- **Library Names**: Filter by library (DIMACS, SDPLIB)
- **Dry Run**: Test without database updates

## 🛠️ Configuration

### Core Configuration Files
- **`config/site_config.yaml`**: Site metadata and overview content
- **`config/solver_registry.yaml`**: Solver display names for reports
- **`config/problem_registry.yaml`**: Problem metadata and classification

### Main Entry Point Options
```bash
# Validation
python main.py --validate                    # Verify environment and libraries

# Benchmarking
python main.py --all                         # Complete benchmark
python main.py --benchmark --problem-set external  # External libraries only
python main.py --benchmark --library_names DIMACS  # DIMACS problems only
python main.py --benchmark --problems nb,arch0     # Specific problems

# Reporting
python main.py --report                      # Generate HTML reports

# Testing
python main.py --benchmark --dry-run         # Test without database changes
```

## 🧪 Development

### Architecture Highlights
- **Unified ProblemData**: SeDuMi-like format with first-class cone structure support
- **Modular Loaders**: Separate MAT and DAT loaders for different formats
- **CVXPY Integration**: Unified solver interface supporting multiple backends
- **Version Tracking**: Complete solver backend and Git commit recording
- **Error Resilience**: Continues benchmarking despite individual solver failures

### Adding New Solvers
1. Extend the CVXPY runner with new backend
2. Add solver configuration to `config/solver_registry.yaml`
3. Test with validation framework
4. Update documentation

### Adding New Problems
1. **External Libraries**: Add as Git submodules in `problems/`
2. **Format Support**: Extend loaders in `scripts/data_loaders/python/`
3. **Validation**: Test with `python main.py --validate`

## 📊 System Achievements

### Technical Accomplishments ✅
- **Unified Architecture**: Single ProblemData format for all optimization types
- **External Library Integration**: Professional-grade problem sets (DIMACS + SDPLIB)
- **Comprehensive Solver Support**: 9 major optimization solvers
- **Production-Ready Reporting**: Interactive dashboards suitable for research publication
- **Fair Baseline Benchmarking**: Minimal configuration for unbiased comparison

### Real-World Impact
- **Research-Ready Data**: Complete JSON/CSV exports for external analysis
- **Reproducible Results**: Version tracking and standardized environments
- **Open Science**: Public reporting with comprehensive problem attribution
- **Community Resource**: Accessible benchmarking platform for optimization research

## 📚 Documentation

### User Guides
- **[CLAUDE.md](CLAUDE.md)**: AI development context and project instructions
- **[docs/guides/](docs/guides/)**: Setup, usage, and integration guides

### Development Documentation
- **[docs/development/basic_design.md](docs/development/basic_design.md)**: Project vision and architecture
- **[docs/development/detail_design.md](docs/development/detail_design.md)**: Technical implementation details
- **[docs/development/conventions.md](docs/development/conventions.md)**: Development standards and protocols
- **[docs/development/tasks.md](docs/development/tasks.md)**: Current development tasks
- **[docs/development/history.md](docs/development/history.md)**: Complete development history

## 🏆 Production Ready Status

### System Completeness ✅
- **139+ External Problems**: Real-world optimization challenges
- **9 Major Solvers**: Comprehensive coverage across optimization types
- **Professional Reporting**: Publication-ready HTML dashboards
- **CI/CD Integration**: Automated benchmarking and deployment
- **Fair Benchmarking Philosophy**: Unbiased solver comparison using defaults

### Quality Assurance
- **Validation Framework**: Data quality checks and error detection
- **Version Tracking**: Complete reproducibility with Git commit recording
- **Testing Infrastructure**: Dry-run mode and comprehensive validation
- **Documentation**: Complete development history and usage guides

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **External Problem Libraries**: VSDP team for DIMACS and SDPLIB repositories
- **Open-Source Solvers**: SciPy, CVXPY, CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HiGHS
- **Web Technologies**: Bootstrap 5, Chart.js for interactive visualizations
- **CI/CD Platform**: GitHub Actions for automated benchmarking and deployment

---

**Status**: Production Ready ✅ | Fair Baseline Benchmarking System Complete ✅  
**Last Updated**: June 2025  
**Achievement**: Comprehensive optimization solver benchmarking platform with unified architecture

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/napinoco/optimization-solver-benchmark) or check the documentation links above.