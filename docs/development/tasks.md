# Development Tasks - DIMACS Registry & New Solvers

**Phase**: DIMACS Integration & Solver Expansion  
**Priority**: High - Adding DIMACS problem registry and CVXOPT/SDPA solver support  
**Context**: Expanding problem coverage and solver options for comprehensive benchmarking

This document provides a granular step-by-step plan to add all DIMACS problems to the registry and implement CVXOPT/SDPA solver support. Each task is incredibly small, testable, and focuses on one specific concern.

---

## Task Queue (Execute in Order)

### **Task 1: Extract DIMACS Problem Data** ⭐ HIGH PRIORITY
**Objective**: Parse all DIMACS problems from README.md into structured format  
**Context**: Need to extract 47+ problems from README tables for registry

**Steps**:
1. Read `problems/DIMACS/README.md` and identify all problem tables
2. Extract problem names, file paths, and known optimal values from each table
3. Create structured data list with fields: display_name, file_path, file_type, library_name, known_objective_value, for_test_flag
4. Map file paths from README to actual paths in `problems/DIMACS/data/` directory structure

**Test Criteria**:
- [ ] All 47+ DIMACS problems identified and structured
- [ ] File paths correctly mapped to actual directory structure
- [ ] Known objective values preserved where available
- [ ] Small problems marked with `for_test_flag: true`

**Definition of Done**: Complete list of DIMACS problems in structured format ready for YAML insertion

### **Task 2: Add DIMACS Problems to Registry** ⭐ HIGH PRIORITY  
**Objective**: Insert all DIMACS problems into `config/problem_registry.yaml`  
**Context**: Registry currently has only 4 DIMACS problems, need all 47+

**Steps**:
1. Open `config/problem_registry.yaml` 
2. Add all DIMACS problems following existing format pattern
3. Use fields: display_name, file_path, file_type, library_name, known_objective_value (if known), for_test_flag
4. Omit problem_type field as requested
5. Set appropriate for_test_flag values (smaller problems = true)

**Test Criteria**:
- [ ] All DIMACS problems added to registry
- [ ] YAML syntax is valid
- [ ] Fields match specification exactly
- [ ] No problem_type fields included
- [ ] File paths point to correct locations

**Definition of Done**: `config/problem_registry.yaml` contains all DIMACS problems with correct metadata

### **Task 3: Add New Solvers to Registry** ⭐ HIGH PRIORITY
**Objective**: Add CVXOPT and SDPA solver entries to `config/solver_registry.yaml`  
**Context**: Registry follows simple display_name pattern

**Steps**:
1. Open `config/solver_registry.yaml`
2. Add entry for CVXOPT: `cvxpy_cvxopt` with display_name "CVXOPT (via CVXPY)"
3. Add entry for SDPA: `cvxpy_sdpa` with display_name "SDPA (via CVXPY)"  
4. Follow existing naming convention pattern

**Test Criteria**:
- [ ] Both solvers added to registry
- [ ] Display names follow "(via CVXPY)" pattern
- [ ] YAML syntax is valid
- [ ] Keys follow existing naming convention

**Definition of Done**: Registry contains new solver entries ready for implementation

### **Task 4: Analyze Current Solver Implementation Pattern** 
**Objective**: Understand existing solver wrapper architecture for consistent implementation  
**Context**: Need to follow established patterns for new solver wrappers

**Steps**:
1. Examine existing solver implementations in `scripts/solvers/python/`
2. Identify `SolverInterface` abstract base class structure
3. Review how CVXPY integration works in existing solvers
4. Understand error handling, result formatting, and configuration patterns
5. Check how solvers are registered and initialized in the system

**Test Criteria**:
- [ ] SolverInterface requirements understood
- [ ] CVXPY integration pattern identified
- [ ] Result format requirements clear
- [ ] Configuration approach documented

**Definition of Done**: Clear understanding of implementation requirements for new solver wrappers

### **Task 5: Implement CVXOPT Solver Wrapper**
**Objective**: Create CVXOPT solver implementation following existing patterns  
**Context**: CVXOPT is a Python optimization package that integrates with CVXPY

**Steps**:
1. Create new solver file following naming convention
2. Implement SolverInterface abstract methods
3. Configure CVXOPT as CVXPY backend
4. Add appropriate error handling and logging
5. Set minimal configuration parameters (verbose: false)
6. Implement result parsing and status code mapping

**Test Criteria**:
- [ ] Solver implements SolverInterface correctly
- [ ] CVXOPT backend properly configured in CVXPY
- [ ] Error handling follows project patterns
- [ ] Results formatted consistently
- [ ] Logging uses appropriate levels

**Definition of Done**: Working CVXOPT solver wrapper that can be instantiated and run

---

### **Task 6: Implement SDPA Solver Wrapper** 
**Objective**: Create SDPA solver implementation following existing patterns  
**Context**: SDPA is specialized for semidefinite programming

**Steps**:
1. Create new solver file following naming convention
2. Implement SolverInterface abstract methods  
3. Configure SDPA as CVXPY backend
4. Add appropriate error handling and logging
5. Set minimal configuration parameters (verbose: false)
6. Implement result parsing and status code mapping
7. Handle SDPA-specific requirements for SDP problems

**Test Criteria**:
- [ ] Solver implements SolverInterface correctly
- [ ] SDPA backend properly configured in CVXPY
- [ ] SDP problem compatibility verified
- [ ] Error handling follows project patterns
- [ ] Results formatted consistently

**Definition of Done**: Working SDPA solver wrapper optimized for SDP problems

---

### **Task 7: Add Solver Dependencies**
**Objective**: Update requirements files to include CVXOPT and SDPA dependencies  
**Context**: New solvers need proper dependency management

**Steps**:
1. Identify current requirements file structure
2. Add CVXOPT package to appropriate requirements file
3. Add SDPA package (if available via pip) or document installation
4. Verify version compatibility with existing CVXPY installation
5. Update any conda/development environment files if present

**Test Criteria**:
- [ ] CVXOPT added to requirements with appropriate version
- [ ] SDPA dependency properly handled
- [ ] No conflicts with existing dependencies
- [ ] Installation instructions updated if needed

**Definition of Done**: Dependency management properly configured for new solvers

---

### **Task 8: Validate New Solver Integration**
**Objective**: Test new solvers work correctly with validation system  
**Context**: System has built-in validation via `python main.py --validate`

**Steps**:
1. Run validation command to test new solver integration
2. Verify solvers can be instantiated without errors
3. Test with small problems marked `for_test_flag: true`
4. Check that results are properly formatted and stored
5. Validate solver version detection works correctly
6. Ensure no regressions in existing solver functionality

**Test Criteria**:
- [ ] `python main.py --validate` passes with new solvers
- [ ] Both CVXOPT and SDPA solvers initialize correctly
- [ ] Test problems solve successfully where appropriate
- [ ] Results stored in database with proper metadata
- [ ] Existing solvers still function correctly
- [ ] No error messages or warnings for working functionality

**Definition of Done**: Complete validation passes, new solvers operational and integrated

---

## Testing Protocol

**After Each Task**:
1. Validate syntax/format of any modified config files
2. Run `python main.py --validate` to check for integration issues
3. Verify no regressions in existing functionality
4. Check git status for clean state before proceeding

**Final Integration Test**:
```bash
# Full system validation
python main.py --validate

# Test specific problem sets
python main.py --benchmark --problem-set dimacs

# Verify new solvers work
python main.py --benchmark --problem-set external
```

---

## Dependencies & Prerequisites

**Required Files**:
- `problems/DIMACS/README.md` (problem source data)
- `config/problem_registry.yaml` (target registry)
- `config/solver_registry.yaml` (target registry)
- `scripts/solvers/python/` (implementation directory)

**System Requirements**:
- Python 3.12+
- CVXPY installation  
- Access to CVXOPT and SDPA packages
- Existing solver implementation patterns

---

## Success Criteria

**DIMACS Integration**:
- All 47+ DIMACS problems accessible via registry
- Problems correctly categorized with metadata
- File paths resolve to actual problem files

**Solver Expansion**:
- CVXOPT and SDPA solvers fully operational
- Integration follows established patterns
- No regressions in existing functionality
- Dependencies properly managed

**System Validation**:
- `python main.py --validate` passes completely
- New solvers solve appropriate test problems
- Results properly stored and formatted

---

*Task-by-task execution ensures incremental progress with validation at each step*
