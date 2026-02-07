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

### **REQUIRED READING**:

1. **[📋 Basic Design](docs/development/basic_design.md)** - Project vision, design philosophy, system overview
2. **[🏗️ Detailed Design](docs/development/detail_design.md)** - Technical architecture, directory structure, implementation specs
3. **[📝 Development Conventions](docs/development/conventions.md)** - Coding standards and engineering guidelines

---

## 🎯 Current Development Status

**Phase**: Research Tool Complete ✅  
**Achievement**: Comprehensive Benchmark System with External Libraries  
**Status**: 139 problems (DIMACS + SDPLIB) with 11 solvers (9 Python + 2 MATLAB)

**Key Achievement**: Comprehensive optimization solver benchmarking platform with external problem libraries, complete version tracking, and research-grade HTML reports with data export capabilities.

---

## 🔧 Development Environment

- **Platform**: GitHub Actions CI/CD with GitHub Pages deployment
- **Languages**: Python 3.12+, MATLAB R2024+ (optional)
- **Storage**: SQLite database (`database/results.db`)
- **Reports**: `docs/pages/` (Bootstrap 5 + Chart.js)

**For directory structure and architecture details, see [detail_design.md](docs/development/detail_design.md).**

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

## 🔄 Development Principles

- **MVP First**: Implement the simplest solution that meets the requirement
- **YAGNI**: Do not add functionality beyond what is explicitly requested
- **User Confirmation**: Ask before implementing features not explicitly requested
- **Commit on Request**: Only commit changes when user explicitly asks

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

*Last Updated: February 2026*