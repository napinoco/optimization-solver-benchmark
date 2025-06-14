# Optimization Solver Benchmark System - MVP Development Plan

## Overview
This plan breaks down the MVP development into small, testable tasks. Each task focuses on a single concern and has clear acceptance criteria.

---

## Phase 1: Project Foundation (Tasks 1-5)

### Task 1: Initialize Repository Structure
**Objective**: Create the basic repository structure with essential files
**Start**: Empty repository
**End**: Repository with folder structure and basic files

**Steps**:
1. Create root directory structure: `config/`, `scripts/`, `problems/`, `database/`, `templates/`, `tests/`
2. Create `README.md` with project description
3. Create `.gitignore` for Python, databases, and logs
4. Create `requirements/base.txt` with essential dependencies: `sqlite3`, `pyyaml`, `jinja2`

**Test**: Verify all folders exist and `pip install -r requirements/base.txt` works

---

### Task 2: Create Database Schema
**Objective**: Define SQLite database structure for storing benchmark results
**Start**: Empty database folder
**End**: Working database schema with tables

**Steps**:
1. Create `database/schema.sql` with tables:
   - `benchmarks` (id, timestamp, environment_info)
   - `results` (id, benchmark_id, solver_name, problem_name, solve_time, status)
   - `problems` (id, name, problem_class, file_path)
   - `solvers` (id, name, version, environment)
2. Create `scripts/database/__init__.py`
3. Create `scripts/database/models.py` with basic database connection and table creation

**Test**: Run script to create database and verify tables exist using SQLite browser

---

### Task 3: Create Configuration Management
**Objective**: Implement YAML-based configuration loading
**Start**: Empty config folder
**End**: Working configuration system

**Steps**:
1. Create `config/benchmark_config.yaml` with basic settings (timeout, parallel_jobs)
2. Create `config/solvers.yaml` with one Python solver definition (scipy)
3. Create `scripts/utils/__init__.py`
4. Create `scripts/utils/config_loader.py` with function to load YAML configs

**Test**: Write test script that loads config and prints values correctly

---

### Task 4: Implement Basic Logging
**Objective**: Set up structured logging system
**Start**: No logging system
**End**: Working logger that writes to file and console

**Steps**:
1. Create `scripts/utils/logger.py` with configured logger
2. Set up log levels (DEBUG, INFO, WARNING, ERROR)
3. Configure file handler for `logs/benchmark.log`
4. Configure console handler for immediate feedback

**Test**: Import logger and verify log messages appear in both file and console

---

### Task 5: Create Sample Problems
**Objective**: Add minimal test problems for initial development
**Start**: Empty problems folder
**End**: Small test problems for LP and QP

**Steps**:
1. Create `problems/light_set/lp/` folder
2. Add simple LP problem in `.mps` format (2 variables, 2 constraints)
3. Create `problems/light_set/qp/` folder  
4. Add simple QP problem in `.qps` format (2 variables, 1 constraint)
5. Create `problems/problem_registry.yaml` listing these problems

**Test**: Verify files exist and are valid MPS/QPS format (can be parsed)

---

## Phase 2: Core Functionality (Tasks 6-12)

### Task 6: Implement Problem Loader
**Objective**: Create system to load and validate problem files
**Start**: Sample problems exist
**End**: Working problem loader that reads MPS/QPS files

**Steps**:
1. Create `scripts/benchmark/__init__.py`
2. Create `scripts/benchmark/problem_loader.py` 
3. Implement function to read MPS files using scipy.optimize
4. Implement function to read QPS files (basic parser)
5. Add validation to check file format and basic structure

**Test**: Load each sample problem and verify correct parsing (no exceptions, reasonable data structure)

---

### Task 7: Create Environment Information Collector
**Objective**: Gather system information for benchmark reproducibility
**Start**: No environment detection
**End**: Working system that collects OS, CPU, memory info

**Steps**:
1. Create `scripts/benchmark/environment_info.py`
2. Implement function to get OS information (platform.system(), platform.release())
3. Implement function to get CPU information (platform.processor(), CPU count)
4. Implement function to get memory information (using psutil)
5. Return structured dictionary with all environment data

**Test**: Run function and verify it returns dict with OS, CPU, memory information

---

### Task 8: Implement Basic Solver Interface
**Objective**: Create abstraction layer for different solvers
**Start**: No solver interface
**End**: Working interface that can run scipy.optimize solvers

**Steps**:
1. Create `scripts/benchmark/solver_interface.py`
2. Define abstract base class `SolverInterface` with `solve()` method
3. Create `SolverResult` class to standardize output (time, status, objective_value)
4. Implement basic error handling and timeout functionality

**Test**: Create mock solver that returns dummy results, verify interface works

---

### Task 9: Implement SciPy Solver Runner
**Objective**: Create concrete implementation for SciPy solvers
**Start**: Basic solver interface exists
**End**: Working SciPy solver for LP problems

**Steps**:
1. Create `scripts/solvers/python/__init__.py`
2. Create `scripts/solvers/python/scipy_runner.py`
3. Implement `ScipySolver` class inheriting from `SolverInterface`
4. Implement LP solving using `scipy.optimize.linprog`
5. Handle timeouts and exceptions properly

**Test**: Run SciPy solver on sample LP problem, verify it returns valid SolverResult

---

### Task 10: Implement Result Collector
**Objective**: Create system to collect and store benchmark results
**Start**: Solver can run problems
**End**: Results are saved to database

**Steps**:
1. Create `scripts/benchmark/result_collector.py`
2. Implement function to insert benchmark metadata into database
3. Implement function to insert individual results into database
4. Add proper foreign key relationships
5. Include error handling for database operations

**Test**: Run solver, collect results, verify data appears correctly in database

---

### Task 11: Create Basic Benchmark Runner
**Objective**: Orchestrate the entire benchmark process
**Start**: Individual components work separately
**End**: Single script that runs complete benchmark

**Steps**:
1. Create `scripts/benchmark/runner.py`
2. Implement `BenchmarkRunner` class that coordinates all components
3. Add method to run single problem with single solver
4. Add method to run multiple problems sequentially
5. Include progress logging and error handling

**Test**: Run benchmark on all sample problems, verify results in database

---

### Task 12: Add CVXPY Solver Support
**Objective**: Add second solver to validate multi-solver architecture
**Start**: Only SciPy solver works
**End**: Both SciPy and CVXPY solvers work

**Steps**:
1. Add CVXPY to `requirements/python.txt`
2. Create `scripts/solvers/python/cvxpy_runner.py`
3. Implement LP and QP solving using CVXPY
4. Update solver configuration in `config/solvers.yaml`
5. Test with different backends (SCS, OSQP)

**Test**: Run benchmark with both solvers, verify different results are stored

---

## Phase 3: Reporting System (Tasks 13-17)

### Task 13: Create HTML Template System
**Objective**: Set up Jinja2 templates for report generation
**Start**: No reporting system
**End**: Basic HTML template that can display data

**Steps**:
1. Create `templates/base.html` with basic HTML structure
2. Create `templates/dashboard.html` extending base template
3. Add CSS framework (simple embedded CSS)
4. Create basic table structure for displaying results
5. Add title, navigation, and footer sections

**Test**: Render template with dummy data, verify HTML is valid and displays correctly

---

### Task 14: Implement HTML Generator
**Objective**: Create system to generate static HTML reports
**Start**: HTML templates exist
**End**: Working HTML generator that creates reports from database

**Steps**:
1. Create `scripts/reporting/__init__.py`
2. Create `scripts/reporting/html_generator.py`
3. Implement function to query database for latest results
4. Implement function to render template with results data
5. Add functionality to save HTML to `docs/` folder

**Test**: Generate HTML report, verify it contains correct data from database

---

### Task 15: Add Basic Statistics
**Objective**: Calculate simple statistics for the dashboard
**Start**: Raw results only
**End**: Dashboard shows summary statistics

**Steps**:
1. Create `scripts/reporting/statistics.py`
2. Implement functions to calculate:
   - Average solve time per solver
   - Success rate per solver
   - Total problems solved
   - Fastest/slowest solvers
3. Return structured data for template rendering

**Test**: Generate statistics, verify calculations are correct manually

---

### Task 16: Create Main Entry Point
**Objective**: Create single script to run complete benchmark + reporting
**Start**: Separate scripts for benchmark and reporting
**End**: Single command runs everything

**Steps**:
1. Create `main.py` in repository root
2. Add command-line argument parsing (argparse)
3. Add options for: run benchmark, generate report, or both
4. Include proper logging and error handling
5. Add help text and usage examples

**Test**: Run `python main.py --help` and `python main.py --all`, verify complete process works

---

### Task 17: Add Results Validation
**Objective**: Ensure benchmark results are reasonable and catch errors
**Start**: No validation of results
**End**: System validates results and flags issues

**Steps**:
1. Create `scripts/utils/validation.py`
2. Implement checks for:
   - Solve times are positive and reasonable (< timeout)
   - Solver status is valid
   - Objective values are reasonable (not NaN/infinite)
3. Add warnings for suspicious results
4. Include validation in result collection process

**Test**: Run validation on existing results, verify it catches intentionally bad data

---

## Phase 4: GitHub Integration (Tasks 18-20)

### Task 18: Create GitHub Actions Workflow
**Objective**: Automate benchmark execution on GitHub
**Start**: Local execution only
**End**: Working GitHub Actions that runs benchmarks

**Steps**:
1. Create `.github/workflows/benchmark.yml`
2. Set up Python environment with dependencies
3. Add step to run benchmarks
4. Add step to generate reports
5. Configure artifact upload for results

**Test**: Push to GitHub, verify workflow runs successfully and produces artifacts

---

### Task 19: Set Up GitHub Pages Deployment
**Objective**: Automatically publish reports to GitHub Pages
**Start**: Reports generated but not published
**End**: Reports automatically published on every run

**Steps**:
1. Update GitHub Actions workflow to deploy to `gh-pages` branch
2. Configure repository settings for GitHub Pages
3. Ensure generated HTML works correctly when served
4. Add automatic commit of results to keep history

**Test**: Verify reports are accessible via GitHub Pages URL after workflow completion

---

### Task 20: Add Manual Trigger and Configuration
**Objective**: Allow manual workflow execution with parameters
**Start**: Only automatic triggers
**End**: Manual workflow with configurable options

**Steps**:
1. Add `workflow_dispatch` trigger to GitHub Actions
2. Add input parameters for:
   - Which solvers to run
   - Which problems to include
   - Timeout settings
3. Update workflow to use input parameters
4. Add validation for input parameters

**Test**: Manually trigger workflow with different parameters, verify they are respected

---

## Acceptance Criteria for MVP

### Functional Requirements
- [ ] System runs at least 2 different solvers (SciPy, CVXPY)
- [ ] System handles at least 2 problem types (LP, QP)
- [ ] Results are stored in SQLite database with proper schema
- [ ] HTML reports are generated and accessible
- [ ] GitHub Actions runs benchmarks automatically
- [ ] Reports are published to GitHub Pages

### Technical Requirements
- [ ] All code has basic error handling
- [ ] System logs important events and errors
- [ ] Database operations are atomic and safe
- [ ] Configuration is externalized in YAML files
- [ ] Code is modular and extensible

### Quality Requirements
- [ ] Each component can be tested independently
- [ ] System gracefully handles solver failures
- [ ] Reports clearly show which tests passed/failed
- [ ] Documentation explains how to add new solvers/problems
- [ ] Repository structure follows the architecture document

---

## Development Notes

### Dependencies
- Core: Python 3.8+, SQLite, YAML, Jinja2
- Solvers: SciPy, CVXPY, (optionally) OSQP, SCS
- System: psutil for environment info
- Dev: pytest for testing

### File Size Considerations
- Keep initial problems under 1MB each
- Use simple MPS/QPS format for maximum compatibility
- Compress problem files if needed

### Error Handling Strategy
- Solver timeouts should not crash the system
- Invalid problem files should be skipped with warnings
- Database errors should be logged but not stop execution
- Network issues in GitHub Actions should be handled gracefully

### Testing Strategy
- Each task should be independently testable
- Manual testing steps provided for each task
- Integration testing after every few tasks
- End-to-end testing before GitHub Actions setup