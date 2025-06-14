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

**Phase**: Phase 2 Final Release Preparation  
**Current Task**: Documentation restructuring and final polish  
**Priority**: Complete remaining tasks before Phase 2 release

**Key Achievement**: Successfully expanded from 2 to 8+ solvers with multi-backend CVXPY support, SOCP/SDP problem types, and comprehensive data publishing infrastructure.

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
│   ├── solvers/     # Solver implementations (Python, Octave)
│   ├── database/    # Data models and storage
│   └── reporting/   # HTML generation and data publishing
├── problems/        # Benchmark problem files
├── docs/           # Generated reports (GitHub Pages)
└── requirements/   # Python dependencies
```

### Current Solver Coverage
```
Problem Type | Solver Count | Backends
LP          | 6            | SciPy + CLARABEL + SCS + ECOS + OSQP + (CVXPY default)
QP          | 6            | SciPy + CLARABEL + SCS + ECOS + OSQP + (CVXPY default)
SOCP        | 4            | CLARABEL + SCS + ECOS + OSQP
SDP         | 2            | CLARABEL + SCS
```

---

## 🚀 Quick Development Commands

```bash
# Validate environment and configuration
python main.py --validate

# Run complete benchmark and reporting
python main.py --all

# Run benchmark only
python main.py --benchmark

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
- **Database Operations**: Must be atomic with proper error handling
- **Result Validation**: Positive solve times, valid status codes required
- **Logging**: Structured with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Problem Formats**: MPS (LP), QPS (QP), Python modules (SOCP/SDP)

---

## 🛠️ Extension Points

### Adding New Solvers
1. Implement `SolverInterface` abstract base class
2. Add configuration to `config/solvers.yaml`
3. Add dependencies to appropriate requirements file
4. Test with validation framework

### Adding New Problems
1. Place files in appropriate `problems/*/type/` directory
2. Update `problems/problem_registry.yaml`
3. Validate with `python main.py --validate`

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
- **Fair, unbiased solver comparison** through minimal configuration
- **Open data publishing** for research community use  
- **Transparent methodologies** with comprehensive documentation
- **Modular extension** for community contributions
- **Production reliability** with comprehensive testing

---

*This dispatch document provides entry point context only. All implementation details, coding standards, and development protocols are documented in the linked files above.*

*Last Updated: December 2025*