# CLAUDE.md - AI Development Context

This file is a navigation hub: detailed design, architecture, and coding standards live in the linked documents below. Read the relevant ones before starting development work.

## Project Overview

This is an automated benchmark system for optimization solvers (LP, QP, SOCP, SDP) that runs on GitHub Actions, stores results in SQLite, and publishes reports via GitHub Pages.

**Core Mission**: *"Regularly benchmark publicly available solvers and publish the results as data"*

## Essential Documentation

1. **[Basic Design](docs/development/basic_design.md)** - Project vision, design philosophy, system overview
2. **[Detailed Design](docs/development/detail_design.md)** - Technical architecture, directory structure, implementation specs
3. **[Development Conventions](docs/development/conventions.md)** - Coding standards and engineering guidelines

## Development Environment

- **Platform**: GitHub Actions CI/CD with GitHub Pages deployment
- **Languages**: Python 3.12+, MATLAB R2020a+ (optional)
- **Storage**: SQLite database (`database/results.db`)
- **Reports**: `docs/pages/` (static HTML, inline CSS)

For directory structure and architecture details, see [detail_design.md](docs/development/detail_design.md).

## Quick Development Commands

```bash
# Clone with submodules (required for external libraries)
git clone --recursive <repository-url>

# Validate environment and external libraries
python main.py --validate

# Run complete benchmark with all libraries
python main.py --all

# Run specific library benchmarks
python main.py --benchmark --library_names DIMACS
python main.py --benchmark --library_names SDPLIB
python main.py --benchmark --library_names NETLIB

# Generate reports only
python main.py --report

# Install dependencies
pip install -r requirements.txt

# Run tests and lint (dev tools: pip install pytest ruff)
pytest tests/
ruff check scripts/ main.py tests/

# Setup MATLAB solvers (optional)
matlab -batch "setup_matlab_solvers"
```

## Development Principles

- **MVP First**: Implement the simplest solution that meets the requirement
- **YAGNI**: Do not add functionality beyond what is explicitly requested
- **User Confirmation**: Ask before implementing features not explicitly requested
- **Commit on Request**: Only commit changes when the user explicitly asks

## Design Philosophy

Fair baseline benchmarking with minimal configuration, reproducible results, and modular architecture for research applications. See [basic_design.md](docs/development/basic_design.md) for the full principles.

## Implementation Notes

- **Solver Configurations**: Use minimal parameters (primarily `verbose: false`) for fair comparison
- **Database Operations**: SQLite with comprehensive metadata and version tracking
- **Result Validation**: Positive solve times, valid status codes, structure analysis
- **External Libraries**: CVXPY conversion for DIMACS/SDPLIB/NETLIB compatibility
- **Version Tracking**: Complete solver backend and Git commit recording
- **Problem Formats**: SeDuMi .mat (DIMACS), SDPA .dat-s (SDPLIB), MPS (NETLIB)

## Known Issues

- **NETLIB Python MPS loader bug**: `scripts/data_loaders/python/mps_loader.py` does not reorder free variables, producing incorrect objective values for 11 NETLIB problems under Python solvers (MATLAB's `mps_loader.m` is unaffected). Not yet fixed. See [NETLIB_KNOWN_OBJECTIVES.md](docs/guides/NETLIB_KNOWN_OBJECTIVES.md) before trusting Python-solver results on NETLIB or working on the MPS loader.

## Critical Development Constraints

These encode past failures; do not relax them without explicit user instruction.

- **Single `requirements.txt`**: All runtime dependencies are pinned in the one root `requirements.txt` — that file *is* the definition of the reproducible benchmark environment. Do not create a `requirements/` directory or split files (base.txt, python.txt); that structure existed before and was deliberately removed. Dev tools (pytest, ruff) stay out of it and are installed separately in CI.
- **CI must fail when problems exist**: Validation workflows exist to detect issues, not hide them. Do not mask failures with patterns like `|| echo "warning"` or graceful degradation — a masked solver installation failure once made CI green while validating nothing.
- **File structure integrity**: Do not reorganize files without explicit user instruction. Generated reports go in `docs/pages/`, the database in `database/results.db`, YAML configs in `config/`.

## Extension Points

For implementation guidance on adding new solvers and problems, see [detail_design.md](docs/development/detail_design.md).

Quick reference: Python solvers via `scripts/solvers/python/solver_configs.py`, MATLAB solvers via `{solver}_runner.m`, new problems via `problem_registry.yaml`.

---

*Last Updated: June 2026*
