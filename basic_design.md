# Optimization Solver Benchmark System - Complete Requirements Document

## Project Overview
An automated system for executing, collecting, and publishing performance benchmark results for optimization solvers (SDP, SOCP, LP, QP).

## Technical Requirements

### Target Solver Environments
- **Python**: CVXPY, Gurobi, MOSEK, SCS, etc.
- **Octave**: Optimization toolbox equivalent (MATLAB-compatible, free)
- **Future Extension**: MATLAB support (requires license purchase or Batch Licensing Pilot application)

### Data Storage Schema
```json
{
  "solver": "Solver name",
  "version": "Version string",
  "timestamp": "Execution datetime",
  "environment": {
    "os": "Operating system information",
    "cpu": "CPU information", 
    "memory": "Memory capacity"
  },
  "problem": {
    "name": "Problem name",
    "class": "Problem class (SDP/SOCP/LP/QP)",
    "source": "Problem source URL"
  },
  "results": {
    "solve_time": "Computation time (seconds)",
    "duality_gap": "Duality gap",
    "status": "Solution status"
  }
}
```

## Infrastructure and Constraint Analysis

### GitHub Actions Constraints and Solutions
- **Time Limit**: 6 hours per job (start within this limit, expand later if needed)
- **Public Repository**: Unlimited execution (private repos have 2,000 minutes/month limit)
- **Extension Options**: Self-hosted runners, external CI integration, manual execution + result upload

### File Storage Strategy

#### GitHub File Size Limitations
- **Single File**: 100MB limit
- **Repository Total**: 1GB recommended, 5GB warning, 100GB block
- **Git LFS**: 1GB free, additional storage at $5/month/50GB

#### Benchmark Problem File Handling
**Problem Size Estimation:**
- SDPLIB: 92 problems total, estimated several GB (large problems are 10-100MB class)
- MIPLIB/NETLIB: Usually several KB to MB
- Total could be several GB to several tens of GB

**Adoption Strategy (Staged):**
```yaml
Phase 1: Lightweight Problem Set
  - Direct GitHub storage (<100MB)
  - Start with small standard problems
  - Verify operation and complete prototype

Phase 2: Large Problem Support  
  - External storage utilization (AWS S3, Google Cloud Storage)
  - GitHub Releases utilization (up to 2GB per single file)
  - URL management via configuration files
  
Phase 3: Complete Version
  - Reference to official URLs of existing benchmark collections
  - Utilize NETLIB, MIPLIB, SDPLIB, etc.
  - Cost optimization
```

## Architecture Design

### System Configuration
```
GitHub Actions (6-hour limit)
├── Environment Setup (Python + Octave)
├── Benchmark Problem Acquisition (URL/Local)
├── Solver Execution (parallelization support)
├── Result Collection & Validation (JSON format)
├── SQLite Database Update
├── Static Site Generation (HTML/CSS/JS)
└── GitHub Pages Auto-Deploy
```

### Extensibility Design Principles
1. **Modular Design** - Independent addition of solvers, problems, and environments
2. **Configuration-Driven** - Manage benchmark content via YAML/JSON configuration
3. **Result Standardization** - Integrate results from different environments using common data format
4. **Error Resilience** - Design that doesn't halt entire system due to partial failures

## Staged Development Plan

### Phase 1: MVP (Minimum Viable Product)
**Goal**: Verify basic functionality
- **Lightweight Problem Set** - Within GitHub limits (tens of MB)
- **Python Environment Only** - CVXPY + basic solvers
- **SQLite Storage** - Local file-based
- **Basic HTML Display** - Tabular result display

### Phase 2: Feature Extension
**Goal**: Practical benchmark system
- **Octave Support** - Execution in MATLAB-compatible environment
- **External Storage** - Support for large problems
- **Detailed Analysis** - Graph and statistical information display
- **CI Optimization** - Parallel execution and cache utilization

### Phase 3: Complete Version
**Goal**: Production-ready public system
- **Manual Result Integration** - Import local and MATLAB execution results
- **Advanced Visualization** - Interactive dashboard
- **Problem Management** - Complete support for standard benchmark collections
- **Performance Optimization** - Support for large-scale execution

## Technical Choice Rationale

### GitHub Actions Adoption Reasons
- **Rich Free Tier** (public repositories)
- **CI/CD Integration** (code changes → automatic execution)
- **Environment Standardization** (reproducible benchmarks)
- **Extensibility** (self-hosted runners, etc.)

### Octave Adoption Reasons
- **MATLAB Compatibility** (can reuse existing code)
- **Free** (no license fees)
- **CI Support** (executable in GitHub Actions)
- **Easy Migration** (can switch to MATLAB in the future)

### SQLite + Static Site Adoption Reasons
- **Simple** (serverless, maintenance-free)
- **Free** (GitHub Pages free hosting)
- **Fast** (static sites display quickly)
- **Easy Migration** (can change DB or move to dynamic sites later)

## Detailed Consideration Results

### Environment Compatibility
**GitHub Actions + Python/Octave:**
- ✅ Python: Full support - standard availability
- ✅ Octave: Executable - dedicated actions available
- ❌ MATLAB: Technically possible but requires expensive licensing

**MATLAB Licensing Options:**
- Batch Licensing Pilot application required
- High license costs
- Commercial use restrictions possible

### File Storage Solutions Evaluated
**GitHub Direct Storage:**
- Pros: Simple, version controlled, free
- Cons: 100MB single file limit, 1GB repository limit
- Use Case: Phase 1 lightweight problems

**External Storage (S3, GCS):**
- Pros: No size limits, reasonable costs
- Cons: Additional complexity, ongoing costs
- Use Case: Phase 2+ large problems

**Git LFS:**
- Pros: Integrated with Git workflow
- Cons: Limited free tier, can become expensive
- Use Case: Medium-sized problems (Phase 1.5)

**GitHub Releases:**
- Pros: 2GB single file limit, free
- Cons: Manual process, not version controlled
- Use Case: Problem set distribution

### CI/CD Platform Comparison
**GitHub Actions (Selected):**
- Pros: Free for public repos, integrated, standardized environments
- Cons: 6-hour time limit, limited to GitHub ecosystem

**Alternatives Considered:**
- CircleCI: 2,500 credits/month free
- GitLab CI: Good free tier but requires GitLab
- Self-hosted: Maximum control but requires infrastructure

## System Constraints and Limitations

### Current Limitations
1. **Execution Time**: 6 hours maximum per GitHub Actions job
2. **File Storage**: 100MB per file for direct GitHub storage
3. **Concurrent Jobs**: 20 maximum parallel jobs
4. **Solver Licensing**: Commercial solvers require separate licensing

### Scaling Solutions
1. **Time Constraints**: Problem set partitioning, parallel execution, self-hosted runners
2. **Storage Constraints**: External storage, problem URL referencing
3. **Licensing**: Community solvers first, commercial solver support via external execution

## Security and Compliance

### Data Protection
- No sensitive data in benchmark problems
- Environment information anonymization options
- Public result publishing with privacy controls

### License Compliance
- Open-source solver prioritization
- Clear licensing documentation for all components
- Commercial solver integration via user-provided licenses

### Execution Security
- Sandboxed execution environments
- Input validation for problem files
- Secure credential management for external services

## Quality Assurance

### Testing Strategy
- Unit tests for each component
- Integration tests for complete workflows
- End-to-end tests including GitHub Actions
- Performance regression testing

### Monitoring and Alerting
- Execution failure notifications
- Performance degradation alerts
- Resource usage monitoring
- Automated issue creation for failures

### Documentation Requirements
- API documentation for extensibility
- User guides for adding new solvers/problems
- Deployment and maintenance documentation
- Troubleshooting guides

## Success Metrics

### Functional Metrics
- Number of supported solvers
- Number of benchmark problems
- Execution success rate
- Report generation reliability

### Performance Metrics
- Average execution time per problem
- System throughput (problems/hour)
- Resource utilization efficiency
- Error recovery time

### Adoption Metrics
- Number of users/viewers
- Community contributions (new solvers/problems)
- External citations/references
- Feature requests and feedback

## Future Roadmap

### Short-term (3-6 months)
- Complete MVP implementation
- Add Octave support
- Implement external storage
- Optimize GitHub Actions workflows

### Medium-term (6-12 months)
- Add MATLAB support
- Implement advanced analytics
- Support custom problem uploads
- Add solver comparison tools

### Long-term (1-2 years)
- Multi-cloud deployment options
- Real-time execution monitoring
- API for external integrations
- Community solver marketplace

This comprehensive requirements document provides the foundation for building a robust, scalable optimization solver benchmark system that can evolve from a simple MVP to a full-featured platform serving the optimization research community.