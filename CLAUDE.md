# CLAUDE.md - AI Development Context

> **⚠️ CRITICAL INSTRUCTION FOR LLM AGENTS:**  
> **You MUST read the linked documentation below BEFORE starting any development work.**  
> This file serves as a dispatching hub - the actual implementation details, coding standards, and development context are in the linked documents.
>
> **⚠️ AFTER AUTO-COMPACT OPERATIONS:**
> **You MUST re-read CLAUDE.md and all related documentation carefully after any auto-compact operation to ensure you maintain proper context and understanding of the project requirements and constraints.**

---

## Project Overview

This is an automated benchmark system for optimization solvers (LP, QP, SOCP, SDP) that runs on GitHub Actions, stores results in SQLite, and publishes reports via GitHub Pages.

**Core Mission**: *"Regularly benchmark publicly available solvers and publish the results as data"*

---

## 🔗 Essential Documentation Links

> **⚠️ DISPATCH HUB NOTICE:**  
> **CLAUDE.md serves as a navigation hub. All detailed design content is in dedicated documents below.**

### **REQUIRED READING** (Read these documents carefully before any development):

1. **[📋 Basic Design](docs/development/basic_design.md)** - Project vision, design philosophy, and system overview
2. **[🏗️ Detailed Design](docs/development/detail_design.md)** - Complete technical architecture, implementation specifications, and code examples
3. **[📝 Development Conventions](docs/development/conventions.md)** - Coding standards, git protocols, and engineering guidelines
4. **[✅ Current Tasks](docs/development/tasks.md)** - Active development tasks for current phase

### **Quick Reference**:
- **[🚀 Setup Guides](docs/guides/)** - Installation and configuration guides
- **[📊 README.md](README.md)** - Project overview and quick start

> **📖 For system architecture, solver configurations, and implementation details, always refer to the dedicated design documents above rather than this file.**

---

## 🎯 Current Development Status

**Phase**: Research Tool Complete ✅  
**Achievement**: Comprehensive Benchmark System with External Libraries  
**Status**: 139 problems (DIMACS + SDPLIB) with 11 solvers (9 Python + 2 MATLAB)

**Key Achievement**: Comprehensive optimization solver benchmarking platform with external problem libraries, complete version tracking, and research-grade HTML reports with data export capabilities.

---

## 🔧 Development Environment Context

### Core Architecture
- **Platform**: GitHub Actions CI/CD with GitHub Pages deployment
- **Languages**: Python 3.12+, MATLAB R2024+
- **Storage**: SQLite database with structured schema
- **Reporting**: Bootstrap 5 + Chart.js interactive dashboards

### Key Directories
```
├── config/          # YAML configuration files
├── scripts/         # Core system implementation
│   ├── benchmark/   # Benchmark execution engine
│   ├── solvers/     # Solver implementations
│   │   ├── python/  # Python solver interfaces
│   │   └── matlab/ # MATLAB interfaces
│   ├── data_loaders/ # Problem format loaders
│   │   ├── python/  # Python loaders (MAT/DAT)
│   │   └── matlab/ # MATLAB loaders
│   ├── utils/       # Problem structure analysis, version detection
│   ├── database/    # Data models and storage
│   └── reporting/   # HTML generation and data publishing
├── problems/        # Benchmark problem files
│   ├── DIMACS/     # External DIMACS library (47 problems)
│   └── SDPLIB/     # External SDPLIB library (92 problems)
├── database/       # SQLite database files
│   └── results.db  # Benchmark results storage
├── docs/           # Generated reports (GitHub Pages)
└── requirements.txt # Python dependencies (single file)
```

### Current Capabilities
**System Status**: Research Tool Complete with **11 solvers** (9 Python + 2 MATLAB) across **139 problems** (DIMACS + SDPLIB)

**For detailed solver coverage and problem statistics, see [basic_design.md](docs/development/basic_design.md).**

---

## 🚀 Quick Development Commands

```bash
# Clone with submodules (required for external libraries)
git clone --recursive <repository-url>

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
pip install -r requirements.txt

# Setup MATLAB solvers (optional)
matlab -batch "setup_matlab_solvers"
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

### MVP Development Principles
- **Minimal Viable Product**: Always implement the simplest solution that meets the requirement
- **No Extra Features**: Do not add functionality beyond what is explicitly requested
- **User Confirmation Required**: Before implementing any feature that seems necessary but wasn't explicitly requested, ask the user for confirmation
- **YAGNI Principle**: "You Aren't Gonna Need It" - implement only what is needed right now
- **Incremental Development**: Build the core functionality first, then iterate based on user feedback

---

## 🎯 Design Philosophy

**For complete design principles and technical philosophy, see [basic_design.md](docs/development/basic_design.md).**

Core principles: Fair baseline benchmarking with minimal configuration, reproducible results, and modular architecture for research applications.

---

## ⚠️ Important Implementation Notes

### Core System Requirements
- **Solver Configurations**: Use minimal parameters (primarily `verbose: false`) for fair comparison
- **Database Operations**: SQLite with comprehensive metadata and version tracking
- **Result Validation**: Positive solve times, valid status codes, structure analysis
- **External Libraries**: CVXPY conversion for DIMACS/SDPLIB compatibility
- **Problem Structure Analysis**: Automatic variable/constraint counting and classification
- **Version Tracking**: Complete solver backend and Git commit recording
- **Logging**: Structured with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Problem Formats**: MPS (LP), QPS (QP), Python (SOCP/SDP), SeDuMi .mat (DIMACS), SDPA .dat-s (SDPLIB)

### 🚨 CRITICAL DEVELOPMENT CONSTRAINTS (DO NOT VIOLATE)

#### Dependencies Management
- **NEVER create new requirements files**: The project uses a single `requirements.txt` file at the root
- **DO NOT create requirements/ directory**: All dependencies are managed through the existing `requirements.txt`
- **DO NOT split requirements**: Never create separate base.txt, python.txt, or other requirement files
- **Use existing requirements.txt**: All GitHub Actions workflows must reference the single `requirements.txt`

#### CI/CD Philosophy  
- **CI must fail when problems exist**: Continuous Integration is designed to detect issues, not hide them
- **DO NOT mask solver installation failures**: If a solver cannot be installed, the CI should fail
- **NO graceful degradation in CI**: Avoid `|| echo "warning"` patterns that mask real problems
- **Validate all dependencies**: The purpose of validation workflows is to ensure all required components work

#### File Structure Integrity
- **Respect existing file organization**: Do not reorganize without explicit user instruction
- **Generated reports location**: HTML reports are generated in `docs/pages/` directory
- **Database location**: SQLite database is at `database/results.db`
- **Configuration location**: YAML configs are in `config/` directory

---

## 🛠️ Extension Points

**For complete implementation guidance on adding new solvers and problems, see [detail_design.md](docs/development/detail_design.md).**

Quick reference: Python solvers via `python_interface.py`, MATLAB solvers via `{solver}_runner.m`, new problems via `problem_registry.yaml`.

---

## 📋 For LLM Agents: Pre-Development Checklist

> **🚨 CRITICAL: CLAUDE.md is a dispatch hub only. All design content is in dedicated documents.**

**BEFORE starting any coding task, confirm you have:**

- [ ] Read [@docs/development/basic_design.md](docs/development/basic_design.md) for project vision, design philosophy, and system overview
- [ ] Read [@docs/development/detail_design.md](docs/development/detail_design.md) for complete technical architecture and implementation details
- [ ] Read [@docs/development/conventions.md](docs/development/conventions.md) for coding standards and protocols
- [ ] Reviewed [tasks.md](docs/development/tasks.md) for current development context
- [ ] Understood that all design decisions and technical details are documented in the linked design files, not in CLAUDE.md

**🚨 CRITICAL CONSTRAINTS CHECKLIST (MUST VERIFY):**

- [ ] **Requirements Management**: Confirmed to use ONLY the existing `requirements.txt` (do not create requirements/ directory)
- [ ] **CI/CD Philosophy**: Understood that CI should fail when problems exist (no graceful degradation)
- [ ] **File Structure**: Verified existing file locations (docs/pages/, database/, config/) before modification
- [ ] **GitHub Workflows**: Ensured any workflow changes reference existing `requirements.txt` file
- [ ] **Solver Dependencies**: Acknowledged that solver installation failures should cause CI to fail

**Failure to read these documents or violating critical constraints will result in implementation that doesn't align with project standards and philosophy.**

---


## 📝 Development Memories

### Task Management
- **Reflection Note**: Please reflect the latest situation into task.md after completing each task.

*This dispatch document provides entry point context only. All implementation details, coding standards, and development protocols are documented in the linked files above.*

*Last Updated: July 2025*