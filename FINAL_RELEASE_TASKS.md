# Final Release Tasks - Phase 2 Branch

This document contains the detailed task breakdown for finalizing the phase2 branch before the first release.

## Task Execution Protocol

1. **Complete one task at a time** following the order below
2. **Test the implementation** using the provided test criteria
3. **Stop and wait** for user review after each task
4. **Commit changes** only after user confirmation
5. **Proceed to next task** once approved

---

## High Priority Tasks (Critical)

### Task 51: Remove QSQP References from Requirements
**Objective**: Remove the non-existent `qsqp>=0.1.0` package from `requirements/python.txt`

**Implementation**:
- Edit `requirements/python.txt`
- Remove line containing `qsqp>=0.1.0`
- Ensure the file remains well-formatted

**Test Criteria**:
- [ ] `qsqp` is completely removed from `requirements/python.txt`
- [ ] File has proper formatting and comments
- [ ] Can install requirements without qsqp errors: `pip install -r requirements/python.txt`

**Files Modified**: `requirements/python.txt`

---

### Task 52: Search and Remove QSQP References from Codebase
**Objective**: Find and remove all references to `qsqp` from Python code files

**Implementation**:
- Search entire codebase for `qsqp` references
- Remove or comment out qsqp-related code in:
  - `scripts/solvers/python/cvxpy_runner.py`
  - Any other files containing qsqp references
- Update solver capability lists if needed

**Test Criteria**:
- [ ] No remaining `qsqp` references in Python files: `grep -r "qsqp" scripts/`
- [ ] CVXPY solver still works without qsqp backend
- [ ] No import errors when running: `python scripts/solvers/python/cvxpy_runner.py`

**Files Modified**: `scripts/solvers/python/cvxpy_runner.py`, potentially others

---

## Medium Priority Tasks (Important)

### Task 53: Standardize Python Versions Between Workflows
**Objective**: Align Python versions used in `octave_test.yml` and `benchmark.yml`

**Implementation**:
- Check current Python versions in both workflows
- Standardize to same version (likely 3.12 based on benchmark.yml)
- Update `octave_test.yml` to match `benchmark.yml`

**Test Criteria**:
- [ ] Both workflows use identical Python version
- [ ] Octave workflow runs successfully with new Python version
- [ ] No compatibility issues introduced

**Files Modified**: `.github/workflows/octave_test.yml`

---

### Task 54: Add Controllable Arguments to Octave Test Workflow
**Objective**: Make `octave_test.yml` consistent with `benchmark.yml` argument pattern

**Implementation**:
- Add `workflow_dispatch` inputs to `octave_test.yml`:
  - `timeout` (solver timeout)
  - `verbose_logging` (enable debug output)
  - `skip_integration_tests` (skip long-running tests)
- Update workflow steps to use these inputs
- Follow same pattern as `benchmark.yml`

**Test Criteria**:
- [ ] Manual trigger available with input options
- [ ] Default values work correctly
- [ ] Custom values are properly applied
- [ ] Workflow remains backward compatible

**Files Modified**: `.github/workflows/octave_test.yml`

---

### Task 55: Evaluate Parallel Jobs Setting
**Objective**: Assess if `parallel_jobs: 2` causes resource contention in benchmarks

**Implementation**:
- Review `config/benchmark_config.yaml`
- Analyze GitHub Actions runner resources (2 CPU cores)
- Consider changing to `parallel_jobs: 1` for fair comparison
- Add comment explaining the choice

**Test Criteria**:
- [ ] Setting is appropriate for GitHub Actions environment
- [ ] Comment explains reasoning for chosen value
- [ ] No negative impact on benchmark fairness

**Files Modified**: `config/benchmark_config.yaml`

---

### Task 58: Create Wide HTML Table (Problems × Solvers)
**Objective**: Generate HTML table with problems vertically, solvers horizontally, showing runtime/objective/dual gap

**Implementation**:
- Create new function in `scripts/reporting/simple_html_generator.py`
- Generate table with:
  - Rows: Problem names
  - Columns: Solver names  
  - Cells: Runtime, objective value, dual gap
- Enable horizontal scrolling for wide tables
- Add to main HTML pages

**Test Criteria**:
- [ ] Table displays correctly with all solvers and problems
- [ ] Horizontal scrolling works on narrow screens
- [ ] Data is accurately populated from database
- [ ] Table is visually clear and readable

**Files Modified**: `scripts/reporting/simple_html_generator.py`, HTML templates

---

### Task 61: Add Author Information to GitHub Pages
**Objective**: Add author attribution to all generated HTML pages

**Implementation**:
- Add author metadata to HTML templates:
  - Author: Naoki Ito
  - Email: napinoco@gmail.com  
  - URL: https://napinoco.github.io
- Update `simple_html_generator.py` to include author info
- Add to page footer or meta tags

**Test Criteria**:
- [ ] Author information appears on all generated pages
- [ ] Contact information is correctly formatted
- [ ] Links are functional
- [ ] Information is tastefully presented

**Files Modified**: `scripts/reporting/simple_html_generator.py`

---

### Task 62: Reorganize Root Directory Markdown Files
**Objective**: Clean up root directory by organizing documentation files

**Implementation**:
- Move feature-specific guides to appropriate folders or `docs/`
- Consolidate history files into single document
- Create unified design document or keep separate as appropriate
- Update `README.md` with links to reorganized files
- Ensure `CLAUDE.md` and `architecture.md` are consistent with current structure

**Test Criteria**:
- [ ] Root directory is cleaner with logical organization
- [ ] All documentation remains accessible
- [ ] Links in README.md work correctly
- [ ] No broken internal references

**Files Modified**: Multiple markdown files, potentially moving several files

---

## Low Priority Tasks (Polish)

### Task 56: Replace Ubuntu Latest with Explicit Version
**Objective**: Specify exact Ubuntu version instead of "Ubuntu Latest"

**Implementation**:
- Check GitHub Actions runner documentation for current Ubuntu version
- Update workflow files to use explicit version (e.g., `ubuntu-22.04`)
- Update any documentation mentioning "Ubuntu Latest"

**Test Criteria**:
- [ ] Workflows use explicit Ubuntu version
- [ ] Version is currently supported by GitHub Actions
- [ ] Documentation is updated accordingly

**Files Modified**: `.github/workflows/*.yml`, documentation files

---

### Task 57: Add Timezone Information to Timestamps
**Objective**: Include timezone data in all recorded timestamps

**Implementation**:
- Update timestamp generation in database operations
- Add timezone info to benchmark results
- Ensure consistent timezone handling across system
- Use UTC as standard timezone

**Test Criteria**:
- [ ] All timestamps include timezone information
- [ ] Timezone is consistently UTC
- [ ] Database schema supports timezone data
- [ ] HTML reports show timezone-aware timestamps

**Files Modified**: `scripts/database/models.py`, timestamp generation code

---

### Task 59: Convert Statistical Analysis Report to HTML
**Objective**: Convert existing statistical analysis reports to HTML format

**Implementation**:
- Locate existing `statistical_analysis_report` functionality
- Create HTML template for statistical reports
- Integrate with existing reporting system
- Ensure proper formatting and readability

**Test Criteria**:
- [ ] Statistical report generates as HTML
- [ ] Data is properly formatted and readable
- [ ] Report integrates with existing HTML pages
- [ ] Statistical calculations are preserved

**Files Modified**: Reporting system files

---

### Task 60: Convert Performance Profiling Report to HTML  
**Objective**: Convert existing performance profiling reports to HTML format

**Implementation**:
- Locate existing `performance_profiling_report` functionality
- Create HTML template for profiling reports
- Integrate with existing reporting system
- Include charts and visualizations if applicable

**Test Criteria**:
- [ ] Performance report generates as HTML
- [ ] Profiling data is clearly presented
- [ ] Report integrates with existing HTML pages
- [ ] Performance metrics are accurately displayed

**Files Modified**: Reporting system files

---

### Task 63: Verify MIT License Appropriateness
**Objective**: Confirm MIT license is appropriate for this benchmarking framework

**Implementation**:
- Research licensing for benchmarking frameworks
- Consider academic/research use cases
- Evaluate if MIT is most appropriate or if another license fits better
- Document reasoning for license choice

**Test Criteria**:
- [ ] License choice is well-researched and appropriate
- [ ] Reasoning is documented
- [ ] License file is properly formatted
- [ ] License is compatible with dependencies

**Files Modified**: `LICENSE`, potentially documentation

---

### Task 64: Implement Preview Hosting for CI
**Objective**: Add temporary hosting during CI for HTML preview before live deployment

**Implementation**:
- Research options: Vercel, Surge, or pages-preview GitHub Action
- Add preview hosting step to GitHub Actions workflow
- Generate preview URL for review
- Ensure preview doesn't interfere with production deployment

**Test Criteria**:
- [ ] Preview URL is generated during CI runs
- [ ] Preview site accurately reflects generated content
- [ ] Preview is accessible for review before live deployment
- [ ] Production deployment remains unaffected

**Files Modified**: `.github/workflows/benchmark.yml`

---

## Testing Protocol

For each task:
1. **Run specific test criteria** listed above
2. **Verify no regressions** in existing functionality  
3. **Test edge cases** where applicable
4. **Check integration** with other system components
5. **Validate output quality** meets standards

## Completion Criteria

All tasks must be completed and tested before the phase2 branch is considered ready for first release. Each task should be committed individually with descriptive commit messages following the established protocol.