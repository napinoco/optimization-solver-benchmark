# Optimization Solver Benchmark System - Basic Design

## Project Vision

Research tool for benchmarking optimization solvers across LP, QP, SOCP, and SDP problems using external problem libraries with automated execution and fair comparison methodology.

---

## Core Mission

**"Regularly benchmark publicly available solvers and publish the results as data"**

This system prioritizes unbiased solver comparison through minimal configuration, establishing "out of the box" performance baselines for research use.

---

## Design Philosophy

### Fair Benchmarking Principles
- **Solver Defaults**: Use default parameters to avoid optimization bias
- **Reproducible Results**: Fixed problem libraries and solver versions
- **Open Data**: Results published as accessible JSON/CSV for research
- **Version Tracking**: Complete metadata for result verification

### Technical Principles
1. **Minimal Configuration**: Fair comparison using solver defaults
2. **Modular Design**: Independent addition of solvers and problems
3. **Result Standardization**: Common data format across solver environments
4. **Error Resilience**: System continues despite individual solver failures
5. **Timeout Management**: Configurable time limits prevent hanging on difficult problems
6. **Automated Operation**: GitHub Actions enables hands-off execution

---

## System Overview

### Deployment & Execution Overview
```
LOCAL DEVELOPMENT:
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

GITHUB ACTIONS (Publishing Only):
┌─────────────┐    ┌──────────────┐
│  Pre-built  │───▶│   GitHub     │
│    docs/    │    │    Pages     │
└─────────────┘    └──────────────┘
```

### Supported Components
**Problem Libraries**:
- **DIMACS**: SeDuMi .mat format
- **SDPLIB**: SDPA .dat-s format
- **NETLIB**: MPS format (LP problems)

Roughly 215 problems are currently registered; the authoritative list (and exact count) is `config/problem_registry.yaml`.

**Solver Support**:
- **Python (9)**: SciPy, CVXPY backends (CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HIGHS)
- **MATLAB (2)**: SeDuMi, SDPT3 (optional)

**Problem Types**:
- **LP**: Linear Programming
- **QP**: Quadratic Programming  
- **SOCP**: Second-Order Cone Programming
- **SDP**: Semidefinite Programming

---

## Implementation Strategy

### Development Approach
**External Library Focus**: Use established problem libraries (DIMACS, SDPLIB, NETLIB) rather than creating internal problems to ensure research relevance and standardization.

**Git Submodule Integration**: Pin problem libraries to specific commits for reproducibility while enabling updates when needed.

**Multi-Language Support**: Unified interface supporting both Python and MATLAB solvers with standardized result format.

### Data Management
**Database Storage**: SQLite with complete version tracking (solver versions, Git commits, environment details) for historical analysis.

**Result Format**: Standardized SolverResult across all solvers with timing, objective values, status, and metadata.

**Report Generation**: Automated HTML dashboards with JSON/CSV data export for research use.

---

## Usage Scenarios

Intended for research and academic use: comparing solver performance across problem types, benchmarking new algorithms against published baselines, and referencing reproducible results in publications.

---

## System Requirements

### Runtime Dependencies
- **Python 3.12+**: Core execution environment
- **Git**: Repository and submodule management
- **Optional MATLAB**: For MATLAB solver support

### Problem Library Setup
```bash
# Clone with submodules for problem libraries
git clone --recursive <repository-url>

# Validate environment
python main.py --validate

# Run benchmarks with default timeout (120s)
python main.py --all

# Run benchmarks with custom timeout for difficult problems
python main.py --all --timeout 600  # 10-minute timeout for large SDP problems
```

---

## Quality Assurance

### Validation Strategy
- **Environment Validation**: System compatibility checking
- **Problem Validation**: File format and data integrity verification  
- **Solver Validation**: Execution capability testing
- **Result Validation**: Output consistency verification
- **Timeout Handling**: Configurable solver execution time limits with graceful termination

### Reproducibility Measures
- **Fixed Dependencies**: Pinned solver versions in requirements.txt
- **Consistent Environments**: Standardized execution via GitHub Actions
- **Complete Metadata**: Environment info saved with all results
- **Version Control**: Git commit tracking for all benchmark runs
- **Original Status Preservation**: Complete solver-specific status codes and statistics preserved for research reproducibility

---

## Future Work

### Potential Enhancements
- **QP/LP Benchmark Expansion**: Add more LP and QP benchmark problems from standard libraries
- **Additional Solver Backends**: Include more solver backends for broader comparison

---

*This basic design establishes the conceptual framework and development approach. For implementation details, see [detail_design.md](detail_design.md).*

*Last Updated: February 2026*