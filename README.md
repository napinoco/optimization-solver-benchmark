# Optimization Solver Benchmark System

A comprehensive open-source platform for benchmarking optimization solvers across multiple problem types (LP, QP, SOCP, SDP). Automated execution via GitHub Actions with interactive HTML reports published to GitHub Pages.

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
- ✅ **Phase 1 Complete**: MVP with Python solvers (SciPy, CVXPY), GitHub Actions CI/CD, interactive reports
- ✅ **Phase 2 Complete**: Multi-backend CVXPY support, external problem libraries, comprehensive reporting
- 🚀 **Production Ready**: Meaningful public reporting system with DIMACS/SDPLIB libraries

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ 
- Git and GitHub account
- Virtual environment recommended

### Installation
```bash
git clone https://github.com/your-username/optimization-solver-benchmark.git
cd optimization-solver-benchmark
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements/base.txt
pip install -r requirements/python.txt
```

### Run Benchmarks
```bash
# Complete benchmark with reports
python main.py --all

# Benchmark specific problem sets
python main.py --benchmark --solvers scipy,cvxpy --problem-set light_set
python main.py --benchmark --solvers cvxpy --problem-set external  # DIMACS/SDPLIB

# Generate reports only  
python main.py --report

# Validate environment and external libraries
python main.py --validate
```

## 🎯 Design Philosophy: Minimal Configuration for Fair Benchmarking

This system prioritizes **fair, unbiased solver comparison** through minimal configuration:

- **Solver Defaults**: Uses each solver's default parameters to avoid optimization bias
- **Baseline Performance**: Establishes "out of the box" performance baselines
- **Transparent Comparison**: Prevents inadvertent parameter tuning that favors specific solvers
- **Future Extensibility**: Advanced parameter optimization available but clearly separated from baseline benchmarks

This approach ensures that benchmarking results reflect genuine solver capabilities rather than configuration expertise.

## 📊 Current Capabilities

### Solvers Supported ✅
- **SciPy**: `linprog` for LP, `quadprog` for QP
- **CVXPY Multi-Backend**: CLARABEL, SCS, ECOS, OSQP with automatic version detection
- **Solver Version Tracking**: Comprehensive version detection and Git commit tracking
- **Fair Baseline Comparison**: Using solver defaults for unbiased benchmarking

### Problem Libraries ✅
**Internal Problems:**
- **LP**: Linear Programming with synthetic test cases
- **QP**: Quadratic Programming with synthetic test cases
- **SOCP**: Portfolio optimization, robust optimization, facility location
- **SDP**: Matrix completion, control theory LMI, Max Cut relaxation

**External Libraries:**
- **DIMACS**: 47 optimization problems in SeDuMi .mat format
- **SDPLIB**: 92 semidefinite programming problems in SDPA format
- **Automatic Structure Analysis**: Variable/constraint counting and problem classification
- **Library Attribution**: Source information displayed in all reports  

### Features ✅
- **Automated Benchmarking**: GitHub Actions CI/CD with manual triggers
- **Interactive Reports**: Bootstrap 5 dashboards with problem structure analysis
- **GitHub Pages**: Automatic report publishing with comprehensive data exports
- **External Library Integration**: DIMACS and SDPLIB via git clone approach
- **Problem Structure Analysis**: Automatic variable/constraint counting and classification
- **Solver Version Tracking**: Backend version detection and Git commit tracking
- **Comprehensive Data Exports**: JSON/CSV files for external consumption
- **Library Source Attribution**: Problem origins clearly displayed in reports
- **Validation Framework**: Data quality checks and error detection
- **Fair Baseline Benchmarking**: Minimal configuration using solver defaults

## 🎯 System Accomplishments ✅

### Complete Solver Ecosystem
- **5 Major Solvers**: CLARABEL, SCS, ECOS, OSQP, SciPy with automatic version detection
- **Solver Version Tracking**: Complete backend version detection and Git commit tracking
- **Fair Baseline Comparison**: Using solver defaults for unbiased benchmarking

### Comprehensive Problem Coverage
- **4 Problem Types**: LP, QP, SOCP, SDP with automatic classification
- **External Libraries**: DIMACS (47 problems) and SDPLIB (92 problems) integrated
- **Problem Structure Analysis**: Automatic variable/constraint counting and classification
- **139 Total Problems**: From synthetic test cases to real-world optimization challenges

### Production-Ready Features
- **Meaningful Public Reporting**: Professional HTML reports with comprehensive data
- **External Library Integration**: Git clone approach for DIMACS/SDPLIB
- **Comprehensive Data Exports**: JSON/CSV files with structure analysis and library attribution
- **Interactive Dashboards**: Bootstrap 5 reports with problem breakdown and solver comparison

## 📁 Project Structure

```
optimization-solver-benchmark/
├── 📋 Core Documentation
│   ├── README.md                # This file
│   ├── architecture.md          # System architecture 
│   └── CLAUDE.md               # AI development context
│
├── 📚 Extended Documentation
│   ├── docs/guides/             # Setup and usage guides
│   └── docs/development/        # Development roadmap and history
│
├── ⚙️  Configuration
│   ├── config/
│   │   ├── benchmark_config.yaml
│   │   └── solvers.yaml
│   └── requirements/
│       ├── base.txt
│       └── python.txt
│
├── 🧪 Core System
│   ├── main.py                  # Entry point
│   ├── scripts/
│   │   ├── benchmark/           # Benchmark execution
│   │   ├── solvers/             # Solver implementations  
│   │   ├── database/            # Data models
│   │   ├── reporting/           # HTML generation
│   │   └── utils/               # Utilities
│   └── templates/               # Jinja2 templates
│
├── 📊 Data & Results  
│   ├── problems/                # Problem files
│   │   ├── DIMACS/             # External DIMACS library (47 problems)
│   │   ├── SDPLIB/             # External SDPLIB library (92 problems)
│   │   └── light_set/          # Internal synthetic problems
│   ├── database/                # SQLite results with version tracking
│   └── docs/                    # Generated HTML reports and data exports
│
└── 🔧 CI/CD
    └── .github/workflows/       # GitHub Actions
```

## 📈 Benchmark Results

Access live benchmark results at: `https://your-username.github.io/optimization-solver-benchmark/`

### Sample Reports
- **Dashboard**: Overall performance metrics with 90+ results from 18 problems
- **Solver Comparison**: Detailed performance breakdown showing success rates and timing
- **Problem Analysis**: Problem structure analysis with variable/constraint counts and library sources
- **Results Matrix**: Comprehensive solver vs problem performance matrix
- **Environment Info**: System specifications with solver versions and Git commit tracking
- **Data Exports**: Complete JSON/CSV exports for external analysis

## 🔧 GitHub Actions Integration

### Automatic Triggers
- **Push to main**: Runs light benchmark set (~2 minutes)
- **Pull requests**: Validation and light testing
- **Scheduled**: Daily comprehensive benchmarks (optional)

### Manual Triggers
Access via GitHub Actions tab → "Run workflow":
- **Solvers**: Choose specific solvers to benchmark
- **Problem Set**: Select from light/medium/large problem sets  
- **Timeout**: Configure solver timeout (10-3600 seconds)
- **Cross-Platform**: Optional testing on Ubuntu/macOS/Windows
- **Verbose Logging**: Enable detailed debug output

## 🛠️ Configuration

### Solver Configuration (`config/solvers.yaml`)
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
    backends: [default]  # Phase 2: [CLARABEL, SCS, ECOS, OSQP, ...]
```

### Benchmark Configuration (`config/benchmark_config.yaml`)
```yaml
benchmark:
  timeout: 300
  problem_sets:
    light_set: "problems/light_set"        # Internal synthetic problems
    external: "problems/external"          # DIMACS/SDPLIB libraries
    dimacs: "problems/DIMACS"             # DIMACS problems only
    sdplib: "problems/SDPLIB"             # SDPLIB problems only
```

## 🧪 Development

### Running Tests
```bash
# Validate environment
python main.py --validate

# Test GitHub Actions simulation
python test_github_actions.py

# Test validation framework
python test_validation.py
```

### Adding New Solvers (Phase 2)
1. Implement solver class inheriting from `SolverInterface`
2. Add solver configuration to `config/solvers.yaml`
3. Add dependencies to appropriate requirements file
4. Test with `python main.py --benchmark --solvers new_solver`

### Adding New Problems
1. **Internal Problems**: Place in `problems/light_set/type/` directory
2. **External Libraries**: Add git submodule or use existing DIMACS/SDPLIB
3. Update `problems/problem_registry.yaml` with problem metadata
4. Validate with `python main.py --validate`

## 📊 Performance Metrics

### Current Solver Coverage ✅
```
Problem Type | SciPy | CLARABEL | SCS | ECOS | OSQP | Total Problems
LP           |   ✅   |    ✅     | ✅  |  ✅   |  ✅   | 12 results
QP           |   ✅   |    ✅     | ✅  |  ✅   |  ✅   | 6 results  
SOCP         |   ❌   |    ✅     | ✅  |  ✅   |  ❌   | 31 results
SDP          |   ❌   |    ✅     | ✅  |  ❌   |  ❌   | 38 results
```

### Real-World Problem Performance
```
Problem Source | Problems | Success Rate | Top Performer
DIMACS        | 47       | ~40%        | CLARABEL
SDPLIB        | 92       | ~35%        | CLARABEL  
Internal      | 6        | 95%         | Multiple
Total         | 145      | 37.8%       | System Average
```

## 🤝 Contributing

### Development Process
1. **Follow task-based development** from `docs/development/PHASE2_TASKS.md`
2. **Test thoroughly** before submitting changes
3. **Update documentation** for new features
4. **Maintain backward compatibility** with Phase 1

### Pull Request Guidelines
- Include comprehensive tests for new solvers/features
- Update configuration examples
- Add problem examples for new types
- Update relevant documentation

## 📚 Documentation

### Core Documentation
- **[README.md](README.md)**: Project overview and quick start guide
- **[CLAUDE.md](CLAUDE.md)**: AI assistant integration and development context

### Setup & Usage Guides
- **[MANUAL_TRIGGER_GUIDE.md](docs/guides/MANUAL_TRIGGER_GUIDE.md)**: GitHub Actions usage guide
- **[GITHUB_ACTIONS_SETUP.md](docs/guides/GITHUB_ACTIONS_SETUP.md)**: CI/CD setup instructions
- **[PR_PREVIEW_GUIDE.md](docs/guides/PR_PREVIEW_GUIDE.md)**: Pull request preview system usage
- **[EXPORT_GUIDE.md](docs/guides/EXPORT_GUIDE.md)**: Data export and integration guide
- **[EXTERNAL_LIBRARIES.md](docs/guides/EXTERNAL_LIBRARIES.md)**: External library setup guide

### Development Documentation
- **[basic_design.md](docs/development/basic_design.md)**: High-level concepts and project vision
- **[detail_design.md](docs/development/detail_design.md)**: Technical architecture and implementation
- **[conventions.md](docs/development/conventions.md)**: Coding standards and development protocols  
- **[history.md](docs/development/history.md)**: Complete development history (Phases 1-2)
- **[tasks.md](docs/development/tasks.md)**: Current development tasks

## 🏆 Achievements & Recognition

### Complete Implementation Accomplishments ✅
- ✅ **Phase 1 MVP**: 20 tasks with Python solvers, CI/CD, interactive reports
- ✅ **Phase 2 Enhanced**: Multi-backend CVXPY, SOCP/SDP support, external libraries
- ✅ **5 Major Solvers**: CLARABEL, SCS, ECOS, OSQP, SciPy with version tracking
- ✅ **139 Total Problems**: Internal synthetic + DIMACS (47) + SDPLIB (92)
- ✅ **Meaningful Public Reporting**: Professional HTML with structure analysis
- ✅ **Production Ready**: Comprehensive data exports, library attribution, Git tracking

### System Impact
- 🎯 **Real-World Problems**: DIMACS and SDPLIB optimization libraries
- 🎯 **Comprehensive Analysis**: Automatic problem structure analysis and classification
- 🎯 **Reproducibility**: Solver version tracking and Git commit recording
- 🎯 **Public Reporting**: Professional dashboards suitable for research publication

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Open-Source Solvers**: SciPy, CVXPY, CLARABEL, SCS, ECOS, OSQP, CBC, HiGHS, GLOP
- **Web Technologies**: Bootstrap 5, Chart.js for interactive visualizations
- **CI/CD Platform**: GitHub Actions for automated benchmarking
- **Community**: Contributors and users providing feedback and improvements

---

**Status**: Production Ready ✅ | Meaningful Public Reporting System Complete ✅  
**Last Updated**: June 2025  
**Achievement**: Comprehensive optimization solver benchmarking platform with external libraries

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-username/optimization-solver-benchmark) or check the documentation above.