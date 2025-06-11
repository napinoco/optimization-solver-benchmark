# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated benchmark system for optimization solvers (SDP, SOCP, LP, QP) that runs on GitHub Actions, stores results in SQLite, and publishes reports via GitHub Pages.

## Architecture

The system follows a modular architecture with these key components:
- **Benchmark Runner**: Orchestrates solver execution across problems
- **Solver Interface**: Abstraction layer for different solvers (Python/CVXPY, SciPy, Octave)
- **Database Layer**: SQLite storage with structured schema for results
- **Reporting System**: HTML generation with Jinja2 templates
- **GitHub Actions**: Automated CI/CD for benchmark execution and deployment

## Key Directories Structure

```
├── config/                    # YAML configuration files
│   ├── benchmark_config.yaml  # Benchmark execution settings
│   ├── solvers.yaml           # Solver definitions and parameters
│   └── environments.yaml      # Environment configurations
├── scripts/
│   ├── benchmark/             # Core benchmark execution engine
│   ├── solvers/               # Solver-specific implementations
│   │   ├── python/            # Python solvers (CVXPY, SciPy, Gurobi)
│   │   └── octave/            # Octave/MATLAB-compatible solvers
│   ├── database/              # Database models and operations
│   ├── reporting/             # HTML/chart generation
│   └── utils/                 # Configuration loading, logging, validation
├── problems/                  # Benchmark problem files
│   ├── light_set/             # Phase 1: Small problems for GitHub storage
│   └── standard_set/          # Phase 2: References to external storage
├── templates/                 # Jinja2 HTML templates
├── database/                  # SQLite database files
└── docs/                      # Generated static site for GitHub Pages
```

## Development Commands

Based on the task structure, the primary development workflow is:

```bash
# Run complete benchmark and reporting
python main.py --all

# Run benchmark only
python main.py --benchmark

# Generate reports only  
python main.py --report

# Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/python.txt  # For Python solvers
pip install -r requirements/dev.txt     # For development
```

## Database Schema

The system uses SQLite with these core tables:
- `benchmarks`: Execution metadata (timestamp, environment_info)
- `results`: Individual solver results (solve_time, status, objective_value)
- `problems`: Problem metadata (name, class, file_path)
- `solvers`: Solver information (name, version, environment)

## Configuration System

All configuration is YAML-based:
- **benchmark_config.yaml**: Timeout, parallel execution settings
- **solvers.yaml**: Solver definitions with installation and execution parameters
- Problem registry files define available benchmark problems

## Solver Integration

New solvers implement the `SolverInterface` abstract base class:
- Must return standardized `SolverResult` objects
- Should handle timeouts and exceptions gracefully
- Results include: solve_time, status, objective_value, duality_gap

## Testing Strategy

Each component should be independently testable:
- Unit tests for individual modules
- Integration tests for complete workflows
- Manual testing steps provided in tasks.md
- End-to-end testing before GitHub Actions deployment

## GitHub Actions Workflow

The system operates under GitHub Actions constraints:
- 6-hour execution limit per job
- Staged development approach (lightweight → full problem sets)
- Artifact handling for database and report files
- Automatic deployment to GitHub Pages

## Development Phases

The project follows a staged development approach:
1. **Phase 1 (MVP)**: ✅ COMPLETED - Python solvers, lightweight problems, basic reporting (Tasks 1-20)
2. **Phase 2**: 🚧 IN PROGRESS - Multi-backend CVXPY, SOCP/SDP support, advanced analytics (Tasks 21-35)
   - ✅ Tasks 21-25: Multi-backend CVXPY support with SOCP and SDP implementation
   - 🚧 Tasks 26-35: Problem classification, external storage, advanced analytics
3. **Phase 3**: PLANNED - Octave support, external storage, advanced problem types
4. **Phase 4**: PLANNED - Production features, cloud deployment, enterprise capabilities

## Important Implementation Notes

- **Minimal Configuration Principle**: Solver configurations use minimal parameters (primarily `verbose: false`) to ensure fair baseline comparison without optimization bias
- Configuration is externalized to YAML files for easy modification
- All database operations should be atomic and include proper error handling
- System must gracefully handle solver failures without stopping execution
- Results require validation (positive solve times, valid status codes)
- Logging is structured with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Problem files use standard formats (MPS for LP, QPS for QP, Python modules for SOCP/SDP)

## CODING PROTOCOL

### Task Development Workflow
**Phase 1**: ✅ COMPLETED (tasks.md → PHASE1_HISTORY.md)
**Phase 2**: Follow PHASE2_TASKS.md sequentially, completing one task at a time:
1. **Complete the task** following its specific requirements in PHASE2_TASKS.md
2. **Test the implementation** using the provided test criteria  
3. **COMMIT THE CHANGES** immediately after successful completion
4. **Stop and wait** for user approval before proceeding to next task

### Coding Instructions
- Write the absolute minimum code required
- No sweeping changes
- No unrelated edits - focus on just the task you're on
- Make code precise, modular, testable
- Don't break existing functionality
- If I need to do anything (e.g. Supabase/AWS config), tell me clearly

### Git Commit Protocol
After each completed task:
- Run `git status` and `git diff` to review changes
- Add relevant files with `git add`
- Create descriptive commit message following the pattern:
  ```
  Complete Task X: Brief description
  
  - Bullet points of what was implemented
  - Key functionality added
  - Any important technical details
  
  🤖 Generated with [Claude Code](https://claude.ai/code)
  
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
