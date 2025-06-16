# CLAUDE.md - AI Development Context

> **⚠️ CRITICAL INSTRUCTION FOR LLM AGENTS:**  
> **You MUST read the linked documentation below BEFORE starting any development work.**  
> This file serves as a dispatching hub - the actual implementation details, coding standards, and development context are in the linked documents.

---

## Project Overview

This is an automated benchmark system for optimization solvers (LP, QP, SOCP, SDP) that runs on GitHub Actions, stores results in SQLite, and publishes reports via GitHub Pages.

**Core Mission**: *"Regularly benchmark publicly available solvers and publish the results as data"*

---

## 🔗 Essential Documentation Links

### **REQUIRED READING** (Read these documents carefully before any development):

1. **[📋 Basic Design](docs/development/basic_design.md)** - High-level concepts, project vision, and development roadmap
2. **[🏗️ Detailed Design](docs/development/detail_design.md)** - Complete technical architecture and implementation specifications  
3. **[📝 Development Conventions](docs/development/conventions.md)** - Coding standards, git protocols, and engineering guidelines
4. **[✅ Current Tasks](docs/development/tasks.md)** - Active development tasks for current phase
5. **[📚 Development History](docs/development/history.md)** - Complete record of completed development phases

### **Quick Reference**:
- **[🚀 Setup Guides](docs/guides/)** - Installation and configuration guides
- **[📊 README.md](README.md)** - Project overview and quick start

---

## 🎯 Current Development Status

**Phase**: Production Ready ✅ | All Sprints Completed Successfully  
**Achievement**: Meaningful Public Reporting System Complete  
**Status**: 139 problems (DIMACS + SDPLIB + Internal) with 5 major solvers

**Key Achievement**: Complete transformation from basic benchmark tool to production-ready public reporting platform with external problem libraries, comprehensive solver version tracking, and professional-grade HTML reports suitable for research publication.

---

## 🔧 Development Environment Context

### Core Architecture
- **Platform**: GitHub Actions CI/CD with GitHub Pages deployment
- **Languages**: Python 3.12+, Octave (MATLAB-compatible)
- **Storage**: SQLite database with structured schema
- **Reporting**: Bootstrap 5 + Chart.js interactive dashboards

### Key Directories
```
├── config/          # YAML configuration files
├── scripts/         # Core system implementation
│   ├── benchmark/   # Benchmark execution engine
│   ├── solvers/     # Solver implementations (Python)
│   ├── external/    # External library loaders (DIMACS, SDPLIB)
│   ├── utils/       # Problem structure analysis, version detection
│   ├── database/    # Data models and storage
│   └── reporting/   # HTML generation and data publishing
├── problems/        # Benchmark problem files
│   ├── DIMACS/     # External DIMACS library (47 problems)
│   ├── SDPLIB/     # External SDPLIB library (92 problems)
│   └── light_set/  # Internal synthetic problems
├── docs/           # Generated reports (GitHub Pages)
└── requirements/   # Python dependencies
```

### Current System Capabilities
```
Problem Type | Total Results | Success Rate | Solver Coverage
LP           | 12 results   | 100%        | SciPy + CLARABEL + SCS + ECOS + OSQP
QP           | 6 results    | 100%        | SciPy + CLARABEL + SCS + ECOS + OSQP  
SOCP         | 31 results   | ~43%        | CLARABEL + SCS + ECOS
SDP          | 38 results   | ~29%        | CLARABEL + SCS
```

### External Problem Libraries
```
Library  | Problems | Format        | Source
DIMACS   | 47       | SeDuMi .mat   | Optimization challenges
SDPLIB   | 92       | SDPA .dat-s   | Semidefinite programming
Internal | 6        | Python        | Synthetic test cases
Total    | 145      | Mixed         | Complete coverage
```

---

## 🚀 Quick Development Commands

```bash
# Validate environment and external libraries
python main.py --validate

# Run complete benchmark with all libraries
python main.py --all

# Run external library benchmarks
python main.py --benchmark --problem-set external

# Run specific library benchmarks
python main.py --benchmark --problem-set dimacs
python main.py --benchmark --problem-set sdplib

# Generate reports only  
python main.py --report

# Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/python.txt
```

---

## 🔄 Development Workflow

### **MANDATORY WORKFLOW** (Follow exactly):
1. **Read Documentation**: Study [basic_design.md](docs/development/basic_design.md), [detail_design.md](docs/development/detail_design.md), and [conventions.md](docs/development/conventions.md)
2. **Check Current Tasks**: Review [tasks.md](docs/development/tasks.md) for active development tasks
3. **Follow Task Protocol**: Complete one task at a time following priority order
4. **Test Implementation**: Validate using task-specific test criteria
5. **Commit Changes**: Use established git commit protocol after user confirmation

### Task-Based Development
- **Sequential Execution**: Complete one task at a time
- **Test-Driven Validation**: Each task includes specific test criteria  
- **Documentation-First**: Update docs for all features
- **User Review**: Stop and wait for approval after each task

---

## 🎯 Design Philosophy

### Fair Baseline Benchmarking
- **Minimal Configuration**: Use solver defaults to avoid optimization bias
- **Transparent Comparison**: Prevent parameter tuning favoring specific solvers
- **Open Data**: Publish results as accessible JSON/CSV for research use
- **Reproducible Results**: Standardized environments ensure consistency

### Technical Principles
- **Modular Design**: Independent solver/problem additions
- **Configuration-Driven**: YAML-based management
- **Error Resilience**: Continue despite individual failures
- **Automated Operation**: GitHub Actions enables hands-off execution

---

## ⚠️ Important Implementation Notes

- **Solver Configurations**: Use minimal parameters (primarily `verbose: false`) for fair comparison
- **Database Operations**: SQLite with comprehensive metadata and version tracking
- **Result Validation**: Positive solve times, valid status codes, structure analysis
- **External Libraries**: CVXPY conversion for DIMACS/SDPLIB compatibility
- **Problem Structure Analysis**: Automatic variable/constraint counting and classification
- **Version Tracking**: Complete solver backend and Git commit recording
- **Logging**: Structured with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Problem Formats**: MPS (LP), QPS (QP), Python (SOCP/SDP), SeDuMi .mat (DIMACS), SDPA .dat-s (SDPLIB)

---

## 🛠️ Extension Points

### Adding New Solvers
1. Implement `SolverInterface` abstract base class
2. Add configuration to `config/solvers.yaml`
3. Add dependencies to appropriate requirements file
4. Test with validation framework

### Adding New Problems
1. **Internal Problems**: Place in `problems/light_set/type/` directory
2. **External Libraries**: Add git submodule or extend existing DIMACS/SDPLIB
3. Update `problems/problem_registry.yaml` with problem metadata
4. Implement loader in `scripts/external/` if new format required
5. Validate with `python main.py --validate`

---

## 📋 For LLM Agents: Pre-Development Checklist

**BEFORE starting any coding task, confirm you have:**

- [ ] Read [basic_design.md](docs/development/basic_design.md) for project vision and goals
- [ ] Read [detail_design.md](docs/development/detail_design.md) for technical architecture
- [ ] Read [conventions.md](docs/development/conventions.md) for coding standards and protocols
- [ ] Reviewed [tasks.md](docs/development/tasks.md) for current development context
- [ ] Understood the fair benchmarking philosophy and minimal configuration approach
- [ ] Familiarized yourself with the modular architecture and extension points

**Failure to read these documents will result in implementation that doesn't align with project standards and philosophy.**

---

## 🤝 Integration Context

This system prioritizes:
- **Fair, unbiased solver comparison** through minimal configuration and real-world problems
- **Meaningful public reporting** suitable for research publication and external evaluation
- **External library integration** with DIMACS and SDPLIB optimization problem sets
- **Comprehensive metadata tracking** including solver versions and Git commit recording
- **Professional-grade reporting** with problem structure analysis and library attribution
- **Open data publishing** with complete JSON/CSV exports for research community use
- **Production reliability** with comprehensive testing across 139 problems

---

*This dispatch document provides entry point context only. All implementation details, coding standards, and development protocols are documented in the linked files above.*

*Last Updated: June 2025 - Production Ready Implementation Complete*