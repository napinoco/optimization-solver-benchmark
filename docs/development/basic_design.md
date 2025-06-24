# Optimization Solver Benchmark System - Basic Design

## Project Vision
Create the leading open-source platform for benchmarking optimization solvers across multiple problem types, providing researchers and practitioners with comprehensive performance insights through automated execution, transparent reporting, and fair baseline comparisons.

---

## Core Mission
**"Regularly benchmark publicly available solvers and publish the results as data"**

This system prioritizes unbiased solver comparison through minimal configuration, establishing "out of the box" performance baselines rather than optimized configurations.

---

## Design Philosophy

### Fair Baseline Benchmarking Principles
- **Solver Defaults**: Uses each solver's default parameters to avoid optimization bias
- **Baseline Performance**: Establishes genuine "out of the box" capabilities
- **Transparent Comparison**: Prevents inadvertent parameter tuning favoring specific solvers
- **Reproducible Results**: Standardized environments ensure consistent benchmarking
- **Open Data**: Results published as accessible JSON/CSV for research use

### Technical Principles
1. **Minimal Configuration**: Fair comparison using solver defaults for unbiased benchmarking
2. **Modular Design**: Independent addition of solvers, problems, and environments
3. **Configuration-Driven**: Manage benchmark content via YAML configuration
4. **Result Standardization**: Common data format across different solver environments
5. **Error Resilience**: System continues despite individual solver failures
6. **Automated Operation**: GitHub Actions enables hands-off execution and deployment

---

## System Overview

### Core Architecture (LOCAL DEVELOPMENT FIRST)
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
                                              │
                                              ▼
                                     ┌─────────────┐
                                     │  Commit to  │
                                     │   docs/     │
                                     └─────────────┘

GITHUB ACTIONS (Publishing Only):
┌─────────────┐    ┌──────────────┐
│  Pre-built  │───▶│   GitHub     │
│    docs/    │    │    Pages     │
└─────────────┘    └──────────────┘
```

### Target Environments ✅
- **Python**: CVXPY multi-backend (CLARABEL, SCS, ECOS, OSQP), SciPy optimization suite
- **Version Tracking**: Complete solver backend version detection and Git commit recording
- **External Libraries**: DIMACS and SDPLIB integration via git clone approach

### Supported Problem Types ✅
- **LP**: Linear Programming (DIMACS)
- **QP**: Quadratic Programming (DIMACS)
- **SOCP**: Second-Order Cone Programming (DIMACS)
- **SDP**: Semidefinite Programming (DIMACS + SDPLIB)
- **Total Coverage**: 139+ problems from external libraries only

---

## Infrastructure Constraints & Solutions

### GitHub Actions Platform
- **Advantages**: Free for public repos, integrated CI/CD, standardized environments
- **Constraints**: 6-hour execution limit, 20 parallel jobs max
- **Solutions**: Staged problem sets, efficient parallel execution

### Storage Strategy ✅ IMPLEMENTED
```yaml
✅ External Problem Libraries - COMPLETED
  - DIMACS library: 47 problems via git submodule
  - SDPLIB library: 92+ problems via git submodule  
  - CVXPY conversion for solver compatibility
  
✅ Production Ready: External-Only Focus
  - 139+ total problems from external libraries
  - Professional reporting with structure analysis
  - Comprehensive metadata and version tracking
  - Simplified architecture with MAT/DAT loaders only
```

---

## Development Roadmap

### Phase 1: MVP Foundation ✅ COMPLETED
**Goal**: Verify basic functionality with minimal viable system

**Achievements**:
- Python solvers (SciPy, CVXPY default backends)
- GitHub Actions CI/CD with manual triggers
- Interactive HTML reports via GitHub Pages
- SQLite storage with structured schema
- Comprehensive validation framework

### Phase 2: Data Publishing Platform ✅ COMPLETED
**Goal**: Robust data publishing with expanded solver ecosystem

**Achievements**:
- Multi-backend CVXPY (CLARABEL, SCS, ECOS, OSQP, CVXOPT, SDPA, SCIP, HiGHS)
- Complete SOCP/SDP problem support via external libraries
- Clean JSON/CSV data exports for research use
- External library integration (DIMACS + SDPLIB)
- Production-ready benchmark system

**Current Solver Coverage**:
```
Problem Type | Solver Count | Backends
LP          | 9            | SciPy + CLARABEL + SCS + ECOS + OSQP + CVXOPT + SDPA + SCIP + HiGHS
QP          | 8            | SciPy + CLARABEL + SCS + ECOS + OSQP + CVXOPT + SCIP + HiGHS
SOCP        | 6            | CLARABEL + SCS + ECOS + CVXOPT + SDPA + SCIP
SDP         | 5            | CLARABEL + SCS + CVXOPT + SDPA + SCIP
```

### Phase 3: Architecture Optimization ✅ COMPLETED
**Goal**: Simplified architecture and enhanced testing capabilities

**Achievements**:
- Direct registry iteration (removed helper functions)
- Clean separation of --library_names vs --problems filtering
- Dry-run mode for testing without database pollution
- Memo column for result annotations
- Streamlined codebase focused on external libraries only
- Removed internal synthetic problems for production clarity

### Phase 4: Advanced Ecosystem ⏳ PLANNED
**Goal**: Enhanced analysis capabilities and broader solver support

**Planned Features**:
- MATLAB Optimization Toolbox integration
- Commercial solver support (Gurobi, CPLEX) via user licenses
- Mixed-Integer Programming (MIP) and Nonlinear Programming (NLP)
- Advanced statistical analysis and performance profiling
- Community problem sharing and solver recommendations

### Phase 5: Production Platform ⏳ PLANNED  
**Goal**: Scalable cloud deployment with enterprise features

**Planned Features**:
- Cloud-native deployment (AWS/GCP/Azure)
- Auto-scaling based on benchmark workload
- API with authentication and quotas
- Real-time performance monitoring
- Academic institution partnerships

---

## Success Metrics

### Functional Metrics
- **Solver Coverage**: Target 10+ open-source solvers across 4 problem types
- **Execution Reliability**: 99%+ successful benchmark completion rate
- **Data Quality**: Comprehensive validation with error detection
- **Report Generation**: Automated HTML/JSON/CSV output

### Performance Metrics
- **Execution Speed**: <5 minutes for light problem sets
- **System Throughput**: Efficient parallel solver execution
- **Resource Utilization**: Optimal GitHub Actions usage

### Adoption Metrics
- **Community Usage**: Active viewers and data consumers
- **Research Impact**: Citations and academic references
- **Contribution Rate**: Community-submitted problems and solvers

---

## Quality Assurance

### Testing Strategy
- **Unit Tests**: Individual component validation
- **Integration Tests**: Complete workflow testing
- **End-to-End Tests**: GitHub Actions simulation
- **Performance Regression**: Continuous performance monitoring

### Validation Framework
- **Input Validation**: Problem file format verification
- **Result Validation**: Solver output consistency checks
- **Environment Validation**: System specification recording
- **Configuration Validation**: YAML syntax and semantic validation

---

## Security & Compliance

### Data Protection
- **Public Data Only**: No sensitive information in benchmark problems
- **Environment Anonymization**: Optional system information privacy
- **Result Transparency**: Open publishing with privacy controls

### License Compliance
- **Open-Source First**: Prioritize freely available solvers
- **Clear Documentation**: License information for all components
- **Commercial Integration**: Support user-provided commercial licenses

---

## Future Vision

### Long-term Goals (2-3 years)
- **Leading Platform**: Recognized standard for optimization solver benchmarking
- **Research Community**: 1000+ active users, 500+ contributed problems
- **Academic Impact**: 100+ citations, integration with research workflows
- **Industry Adoption**: Enterprise use cases and professional references

### Technology Evolution
- **Multi-language Support**: Python, Julia, MATLAB, C++ solver integration
- **Advanced Analytics**: Machine learning for solver performance prediction
- **Real-time Monitoring**: Live performance tracking and alerting
- **Cloud Deployment**: Scalable infrastructure with global availability

---

## Project Values

### Open Source Commitment
- **Transparent Development**: Public development process
- **Community-Driven**: User feedback shapes feature development
- **Open Data**: All results publicly available for research
- **Free Access**: Core functionality available without cost

### Scientific Rigor
- **Reproducible Results**: Consistent, repeatable benchmarking methodology
- **Fair Comparison**: Unbiased evaluation using solver defaults
- **Statistical Validity**: Proper performance analysis and reporting
- **Documentation**: Comprehensive methodology and implementation details

---

*This basic design document establishes the high-level vision, principles, and roadmap for the optimization solver benchmark system. For detailed implementation specifications, see [detail_design.md](detail_design.md).*

*Last Updated: December 2025*