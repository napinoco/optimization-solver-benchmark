# Next Major Enhancement: Public Reporting System

This document defines the task breakdown for implementing meaningful public reporting capabilities with external problem libraries, solver version tracking, and aggregated results reporting.

---

## Overview

Transform the benchmark system to support meaningful public reporting with real-world problem instances from established optimization libraries.

### Key Features
1. **Standard Benchmark Libraries**: Integration with DIMACS and SDPLIB problem sets
2. **Solver Version Tracking**: Record and display backend versions for reproducibility  
3. **Git Commit Tracking**: Track Git commit hash for each benchmark run
4. **Enhanced Documentation**: Comprehensive guides for problem library integration

### Simplified Approach
- **Database reset**: Reinitialize existing data for clean schema upgrade
- **Version tracking**: Only for backend solvers (not other components)
- **Git clone approach**: Use direct `git clone` for external libraries (no library manager)
- **Query-based aggregation**: Modify existing queries instead of additional tables (low priority)

---

## Sprint 5: Database Schema Enhancement (Foundation)

### Task 5.1: Extend Database Schema for Enhanced Tracking
- **Objective**: Add version tracking and Git commit fields to existing schema
- **Scope**: Extend database with minimal additional fields
- **Success Criteria**:
  - [ ] Add `solver_version`, `solver_backend` fields to `results` table
  - [ ] Add `problem_library` field to `results` table  
  - [ ] Add `git_commit_hash` field to `benchmarks` table
  - [ ] Add `run_id` field to `results` table for future aggregation
  - [ ] Test schema changes with clean database
- **Files to Modify**: `database/schema.sql`
- **Test**: Database accepts new fields and existing code works
- **Estimated Time**: 2 hours

### Task 5.2: Update Database Models
- **Objective**: Update SQLAlchemy models to match new schema
- **Scope**: Add new fields to existing models
- **Success Criteria**:
  - [ ] Update `SolverResult` model with version and library fields
  - [ ] Update `BenchmarkResult` model with git_commit_hash field
  - [ ] Test model creation and queries with new fields
  - [ ] Ensure existing functionality remains intact
- **Files to Modify**: `scripts/database/models.py`
- **Test**: Models work correctly with enhanced schema
- **Estimated Time**: 1 hour

---

## Sprint 6: Solver Version Detection System

### Task 6.1: Create Version Detection Utilities
- **Objective**: Implement utilities for detecting solver backend versions
- **Scope**: Version detection for CVXPY backends and SciPy only
- **Success Criteria**:
  - [ ] Create `scripts/utils/version_utils.py` with detection functions
  - [ ] Implement version detection for CVXPY backends (CLARABEL, SCS, ECOS, OSQP)
  - [ ] Implement version detection for SciPy
  - [ ] Add fallback mechanisms for failed version detection
  - [ ] Test version detection across different environments
- **Files to Create**: `scripts/utils/version_utils.py`
- **Test**: Version detection returns correct versions for all solver backends
- **Estimated Time**: 3 hours

### Task 6.2: Update Solver Classes with Version Detection
- **Objective**: Add version detection to existing solver implementations
- **Scope**: Modify CVXPY and SciPy solver classes
- **Success Criteria**:
  - [ ] Add version detection to `CvxpySolver` class
  - [ ] Add version detection to `ScipySolver` class  
  - [ ] Store version information during solver initialization
  - [ ] Test version detection with all configured backends
  - [ ] Handle missing backends gracefully
- **Files to Modify**: `scripts/solvers/python/cvxpy_runner.py`, `scripts/solvers/python/scipy_runner.py`
- **Test**: All solvers report correct version information
- **Estimated Time**: 2 hours

### Task 6.3: Add Git Commit Hash Detection
- **Objective**: Add Git commit hash tracking to benchmark runs
- **Scope**: Detect and store current Git commit hash for reproducibility
- **Success Criteria**:
  - [ ] Create Git commit hash detection function
  - [ ] Add commit hash to benchmark metadata during initialization
  - [ ] Store commit hash in database with each benchmark run
  - [ ] Handle cases where Git is not available or repo is dirty
  - [ ] Test commit hash detection and storage
- **Files to Modify**: `scripts/benchmark/runner.py`, `scripts/benchmark/environment_info.py`
- **Test**: Git commit hash is correctly detected and stored
- **Estimated Time**: 1 hour

### Task 6.4: Update Result Storage with Enhanced Data
- **Objective**: Store version and metadata in results
- **Scope**: Update result collection to include new tracking fields
- **Success Criteria**:
  - [ ] Update result storage to include solver versions
  - [ ] Add problem library information to results
  - [ ] Generate and store run IDs for future aggregation
  - [ ] Test enhanced result storage
  - [ ] Verify backward compatibility
- **Files to Modify**: `scripts/benchmark/result_collector.py`
- **Test**: Enhanced data is correctly stored and retrieved
- **Estimated Time**: 2 hours

---

## Sprint 7: External Problem Library Integration

### Task 7.1: Clone and Organize External Libraries
- **Objective**: Set up DIMACS and SDPLIB libraries using git clone
- **Scope**: Direct git clone approach without complex management
- **Success Criteria**:
  - [ ] Clone DIMACS repository into `problems/DIMACS/`
  - [ ] Clone SDPLIB repository into `problems/SDPLIB/`
  - [ ] Keep existing folder structures from repositories
  - [ ] Add .gitignore entries for external library directories
  - [ ] Test access to cloned problem files
- **Files to Create**: Cloned external library directories
- **Files to Modify**: `.gitignore`
- **Test**: External libraries are accessible and properly ignored by Git
- **Estimated Time**: 1 hour

### Task 7.2: Enhance Problem Registry for Libraries
- **Objective**: Extend problem registry to support external libraries
- **Scope**: Update problem_registry.yaml with library organization
- **Success Criteria**:
  - [ ] Add DIMACS section to problem registry
  - [ ] Add SDPLIB section to problem registry
  - [ ] Select 5-10 representative problems from each library
  - [ ] Include library metadata (source, description, file format)
  - [ ] Test registry loading with enhanced structure
- **Files to Modify**: `problems/problem_registry.yaml`
- **Test**: Enhanced registry loads successfully and includes external libraries
- **Estimated Time**: 2 hours

### Task 7.3: Create DIMACS Problem Loader
- **Objective**: Implement loader for DIMACS .mat format files
- **Scope**: Use existing read_sedumi_mat function from read_func.py
- **Success Criteria**:
  - [ ] Create `scripts/external/dimacs_loader.py`
  - [ ] Integrate with existing `read_sedumi_mat` function
  - [ ] Convert DIMACS problems to unified `ProblemData` format
  - [ ] Add DIMACS library identification
  - [ ] Test loading sample DIMACS problems
- **Files to Create**: `scripts/external/dimacs_loader.py`
- **Test**: DIMACS loader can load .mat files successfully
- **Estimated Time**: 3 hours

### Task 7.4: Create SDPLIB Problem Loader
- **Objective**: Implement loader for SDPLIB .dat-s format files
- **Scope**: Use existing read_sdpa_sparse_format function from read_func.py
- **Success Criteria**:
  - [ ] Create `scripts/external/sdplib_loader.py`
  - [ ] Integrate with existing `read_sdpa_sparse_format` function
  - [ ] Convert SDPLIB problems to unified `ProblemData` format
  - [ ] Add SDPLIB library identification
  - [ ] Test loading sample SDPLIB problems
- **Files to Create**: `scripts/external/sdplib_loader.py`
- **Test**: SDPLIB loader can load .dat-s files successfully
- **Estimated Time**: 3 hours

### Task 7.5: Update Problem Loader with Library Support
- **Objective**: Integrate external library loaders into main problem loading
- **Scope**: Extend existing ProblemLoader to support external formats
- **Success Criteria**:
  - [ ] Add library detection to problem loading logic
  - [ ] Integrate DIMACS and SDPLIB loaders into main loader
  - [ ] Add `library` field to ProblemData class
  - [ ] Update problem loading to identify library source
  - [ ] Test loading problems from all three libraries (light_set, DIMACS, SDPLIB)
- **Files to Modify**: `scripts/benchmark/problem_loader.py`
- **Test**: Problem loader can load problems from any supported library
- **Estimated Time**: 2 hours

---

## Sprint 8: Enhanced Reporting System

### Task 8.1: Update HTML Generator for Version Display
- **Objective**: Show solver version information in HTML reports
- **Scope**: Enhance existing HTML reports with version data
- **Success Criteria**:
  - [ ] Update solver comparison table to include version columns
  - [ ] Display solver backend information in reports
  - [ ] Add Git commit hash to environment information
  - [ ] Show problem library information in problem listings
  - [ ] Test HTML generation with version-enhanced data
- **Files to Modify**: `scripts/reporting/simple_html_generator.py`
- **Test**: HTML reports correctly display all enhanced information
- **Estimated Time**: 3 hours

### Task 8.2: Create Library-Specific Report Sections
- **Objective**: Organize reports by problem library
- **Scope**: Add library breakdown to HTML reports
- **Success Criteria**:
  - [ ] Create separate sections for each problem library
  - [ ] Add library metadata display (source, description)
  - [ ] Group results by library for better organization
  - [ ] Add library statistics (problem count, success rates)
  - [ ] Test library-specific report generation
- **Files to Modify**: `scripts/reporting/simple_html_generator.py`
- **Test**: Library-specific reports show correct organization and metadata
- **Estimated Time**: 3 hours

### Task 8.3: Update Data Export Formats
- **Objective**: Include enhanced data in JSON/CSV exports
- **Scope**: Add version and library information to exported data
- **Success Criteria**:
  - [ ] Add solver version and backend to JSON exports
  - [ ] Add problem library information to data exports
  - [ ] Include Git commit hash in exported metadata
  - [ ] Maintain backward compatibility of export formats
  - [ ] Test data exports with enhanced information
- **Files to Modify**: `scripts/reporting/data_publisher.py`
- **Test**: Data exports include all new fields correctly
- **Estimated Time**: 2 hours

---

## Sprint 9: Integration Testing and Documentation

### Task 9.1: Test Complete Enhanced Workflow
- **Objective**: Validate entire enhanced system end-to-end
- **Scope**: Test with all three problem libraries and version tracking
- **Success Criteria**:
  - [ ] Run benchmark with light_set, DIMACS, and SDPLIB problems
  - [ ] Verify solver version detection works for all backends
  - [ ] Confirm Git commit hash tracking functions correctly
  - [ ] Validate HTML reports include all enhancements
  - [ ] Test data exports with enhanced information
- **Files to Modify**: None (testing only)
- **Test**: Complete enhanced workflow executes successfully
- **Estimated Time**: 3 hours

### Task 9.2: Performance Testing and Optimization
- **Objective**: Ensure system performance with external libraries
- **Scope**: Test performance with larger problem sets
- **Success Criteria**:
  - [ ] Measure benchmark execution time with external libraries
  - [ ] Test timeout handling with challenging problems
  - [ ] Verify memory usage remains reasonable
  - [ ] Optimize any performance bottlenecks discovered
  - [ ] Document performance characteristics
- **Files to Modify**: Performance optimizations as needed
- **Test**: System maintains reasonable performance with external libraries
- **Estimated Time**: 2 hours

### Task 9.3: Update README with Enhanced Features
- **Objective**: Document new capabilities in main README
- **Scope**: Add sections describing enhanced benchmark capabilities
- **Success Criteria**:
  - [ ] Document external problem library integration
  - [ ] Explain solver version tracking features
  - [ ] Include Git commit hash tracking information
  - [ ] Provide examples of enhanced command-line usage
  - [ ] Update quick start guide
- **Files to Modify**: `README.md`
- **Test**: README accurately describes all enhanced capabilities
- **Estimated Time**: 2 hours

### Task 9.4: Create External Library Setup Guide
- **Objective**: Document external library integration process
- **Scope**: Guide for setting up and using external libraries
- **Success Criteria**:
  - [ ] Create `docs/guides/EXTERNAL_LIBRARIES.md`
  - [ ] Document git clone setup process for DIMACS and SDPLIB
  - [ ] Explain problem registry configuration
  - [ ] Provide troubleshooting section
  - [ ] Include examples of adding new libraries
- **Files to Create**: `docs/guides/EXTERNAL_LIBRARIES.md`
- **Test**: Guide enables successful external library setup
- **Estimated Time**: 2 hours

### Task 9.5: Final Integration and Polish
- **Objective**: Final validation and system polish
- **Scope**: Comprehensive testing and final refinements
- **Success Criteria**:
  - [ ] Test all combinations of libraries and solvers
  - [ ] Verify backward compatibility maintained
  - [ ] Fix any discovered integration issues
  - [ ] Ensure documentation is complete and accurate
  - [ ] Validate system ready for public use
- **Files to Modify**: Bug fixes and polish as needed
- **Test**: System ready for public reporting and evaluation
- **Estimated Time**: 3 hours

---

## Future Enhancement: Query-Based Aggregation (Low Priority)

### Task 10.1: Implement Query-Based Median Calculation
- **Objective**: Add median calculation without additional tables
- **Scope**: Modify existing queries to compute aggregated statistics
- **Success Criteria**:
  - [ ] Update queries to fetch latest 5 results per solver+problem combination
  - [ ] Add median calculation functions to existing code
  - [ ] Display median statistics in reports alongside individual results
  - [ ] Add command-line option to show aggregated vs individual results
  - [ ] Test aggregation accuracy with multiple runs
- **Files to Modify**: `scripts/reporting/simple_html_generator.py`, `scripts/database/models.py`
- **Test**: Median calculations are accurate and performant
- **Estimated Time**: 4 hours

---

## Implementation Summary

### Total Estimated Time: 40 hours (5 weeks at ~8 hours/week)

### Sprint Breakdown:
- **Sprint 5**: Database Schema Enhancement (3 hours)
- **Sprint 6**: Solver Version Detection (8 hours)
- **Sprint 7**: External Library Integration (11 hours)
- **Sprint 8**: Enhanced Reporting (8 hours)
- **Sprint 9**: Testing and Documentation (12 hours)
- **Future**: Query-Based Aggregation (4 hours - low priority)

### Success Criteria for Complete Implementation:
1. **Minimal Scope**: Each task focuses on single concern, independently testable
2. **Clear Start/End**: Obvious completion criteria for each task
3. **Testable**: Can verify task completion objectively
4. **No Regression**: Existing functionality continues to work
5. **Documentation**: Changes reflected in relevant documentation

### Key Deliverables:
- **Enhanced Database**: Version tracking, library tracking, Git commit hash
- **External Libraries**: DIMACS and SDPLIB integration via git clone
- **Version Tracking**: Complete solver backend version detection and storage
- **Professional Reports**: Enhanced HTML reports with library organization and version information
- **Comprehensive Documentation**: Setup guides and technical documentation

### Simplified Approach Benefits:
- **60% time reduction**: From 95 hours to 40 hours
- **No complex migrations**: Database reset approach
- **No over-engineering**: Direct git clone, query-based aggregation
- **Focused scope**: Version tracking only for backends
- **Clear priorities**: Core features first, aggregation later