# Optimization Solver Benchmark System

A comprehensive open-source platform for benchmarking optimization solvers across multiple problem types (LP, QP, SOCP, SDP). Automated execution via GitHub Actions with interactive HTML reports published to GitHub Pages.

## 🎯 Current Status: Phase 2 Development
- ✅ **Phase 1 Complete**: MVP with Python solvers (SciPy, CVXPY), GitHub Actions CI/CD, interactive reports
- 🚧 **Phase 2 In Progress**: Multi-backend CVXPY support, SOCP/SDP problems, advanced analytics

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

# Benchmark only
python main.py --benchmark --solvers scipy,cvxpy --problem-set light_set

# Generate reports only  
python main.py --report

# Validate environment
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

### Solvers Supported
**Phase 1 ✅ Completed:**
- **SciPy**: `linprog` for LP, `quadprog` for QP
- **CVXPY**: Default backends for LP/QP problems

**Phase 2 🚧 In Progress:**
- **CVXPY Multi-Backend**: CLARABEL, SCS, ECOS, OSQP for expanded solver coverage
- **Minimal Configuration**: Fair baseline comparison using solver defaults for unbiased benchmarking

### Problem Types
**Phase 1 ✅ Completed:**
- **LP**: Linear Programming (.mps format)
- **QP**: Quadratic Programming (.qps format)

**Phase 2 ✅ SOCP & SDP Support Added:**
- **SOCP**: Second-Order Cone Programming with three example problems:
  - Portfolio optimization (Markowitz risk minimization)
  - Robust optimization with ellipsoidal uncertainty
  - Facility location with Euclidean distance constraints
- **SDP**: Semidefinite Programming with three example problems:
  - Matrix completion (nuclear norm minimization)
  - Control theory (Lyapunov stability analysis via LMI)
  - Combinatorial optimization (Max Cut relaxation)  

### Features
- **Automated Benchmarking**: GitHub Actions CI/CD with manual triggers
- **Interactive Reports**: Bootstrap 5 dashboards with Chart.js visualizations
- **GitHub Pages**: Automatic report publishing  
- **Validation Framework**: Data quality checks and error detection
- **Comprehensive Logging**: Structured logging with multiple levels
- **Flexible Configuration**: YAML-based solver and benchmark settings
- **Fair Baseline Benchmarking**: Minimal configuration approach using solver defaults for unbiased comparison

## 🎯 Phase 2 Roadmap (In Progress 🚧)

### Enhanced Solver Ecosystem
- **10+ Open-Source Solvers**: CLARABEL, SCS, ECOS, OSQP, QSQP, CBC, GLOP, HiGHS via CVXPY
- **Intelligent Backend Selection**: Automatic solver selection based on problem characteristics
- **Solver Capability Matrix**: Comprehensive compatibility mapping

### Expanded Problem Types
- **SOCP** ✅: Second-Order Cone Programming with portfolio optimization, robust optimization, and facility location examples
- **SDP** ✅: Semidefinite Programming with matrix completion, control theory, and combinatorial optimization examples
- **Problem Classification** 🚧: Automatic type detection and difficulty estimation

### Advanced Features
- **External Storage**: Support for large problem sets via GitHub releases/URLs
- **Performance Analytics**: Time-series analysis and regression detection  
- **Solver Recommendations**: AI-powered solver selection engine
- **Export Capabilities**: PDF reports, CSV data, JSON API

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
│   ├── database/                # SQLite results
│   └── docs/                    # Generated reports
│
└── 🔧 CI/CD
    └── .github/workflows/       # GitHub Actions
```

## 📈 Benchmark Results

Access live benchmark results at: `https://your-username.github.io/optimization-solver-benchmark/`

### Sample Reports
- **Dashboard**: Overall performance metrics and solver comparison
- **Solver Analysis**: Detailed performance breakdown per solver
- **Problem Analysis**: Problem-specific insights and difficulty ratings
- **Environment Info**: System specifications and reproducibility data

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
    light_set: "problems/light_set"
    medium_set: "problems/medium_set"  
    large_set: "problems/large_set"
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
1. Place problem files in appropriate `problems/*/type/` directory
2. Update `problems/problem_registry.yaml`
3. Validate with `python main.py --validate`

## 📊 Performance Metrics

### Solver Comparison Matrix (Phase 1)
```
Problem Type | SciPy | CVXPY | Status
LP           |   ✅   |   ✅   | Production Ready
QP           |   ✅   |   ✅   | Production Ready  
SOCP         |   ❌   |   🚧   | Phase 2 Development
SDP          |   ❌   |   🚧   | Phase 2 Development
```

### Expected Phase 2 Coverage
```
Problem Type | Solver Count | Best Performance
LP           | 7 solvers    | HiGHS, CBC, CLARABEL
QP           | 6 solvers    | OSQP, CLARABEL, SCS
SOCP         | 4 solvers    | CLARABEL, ECOS, SCS  
SDP          | 2 solvers    | CLARABEL, SCS
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
- **[EXTERNAL_STORAGE.md](docs/guides/EXTERNAL_STORAGE.md)**: External storage configuration
- **[OCTAVE_INTEGRATION.md](docs/guides/OCTAVE_INTEGRATION.md)**: Octave solver integration

### Development Documentation
- **[basic_design.md](docs/development/basic_design.md)**: High-level concepts and project vision
- **[detail_design.md](docs/development/detail_design.md)**: Technical architecture and implementation
- **[conventions.md](docs/development/conventions.md)**: Coding standards and development protocols  
- **[history.md](docs/development/history.md)**: Complete development history (Phases 1-2)
- **[tasks.md](docs/development/tasks.md)**: Current development tasks

## 🏆 Achievements & Recognition

### Phase 1 Accomplishments
- ✅ **Complete MVP** with 20 tasks successfully implemented
- ✅ **Production CI/CD** with GitHub Actions and Pages deployment
- ✅ **Interactive Reports** with professional Bootstrap dashboards
- ✅ **Comprehensive Documentation** with setup guides and user manuals
- ✅ **Validation Framework** ensuring data quality and reliability

### Phase 2 Goals
- 🎯 **10x Solver Coverage**: Expand from 2 to 10+ solver implementations
- 🎯 **4x Problem Types**: Add SOCP and SDP to existing LP/QP support
- 🎯 **Advanced Analytics**: Performance tracking and intelligent recommendations
- 🎯 **Production Ready**: Containerization, monitoring, and scalability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Open-Source Solvers**: SciPy, CVXPY, CLARABEL, SCS, ECOS, OSQP, CBC, HiGHS, GLOP
- **Web Technologies**: Bootstrap 5, Chart.js for interactive visualizations
- **CI/CD Platform**: GitHub Actions for automated benchmarking
- **Community**: Contributors and users providing feedback and improvements

---

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🚧  
**Last Updated**: December 2025  
**Next Milestone**: Task 21 - CVXPY Multi-Backend Support

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-username/optimization-solver-benchmark) or check the documentation above.