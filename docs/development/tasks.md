# Simplification Tasks for Reliable Deployment

This document outlines specific tasks to simplify the optimization solver benchmark system and achieve a working, publishable state.

---

## Goal: Working and Publishable State

**Key Principles:**
- Simplify architecture to reduce failure points
- Run benchmarks locally, use GitHub Actions only for publishing
- Remove Octave support temporarily
- Clean up test and unused code
- Focus on reliability over features

---

## Phase 1: Architecture Simplification (High Priority)

### Task 1.1: Remove Octave Support
- **Objective**: Eliminate Octave integration to reduce complexity
- **Scope**: Remove all Octave-related code and configurations
- **Success Criteria**:
  - [ ] Delete `scripts/solvers/octave/` directory
  - [ ] Remove Octave references from `config/solvers.yaml`
  - [ ] Delete `config/octave_config.yaml`
  - [ ] Remove `.github/workflows/octave_test.yml`
  - [ ] Update documentation to reflect Python-only approach
- **Files to Modify**: Multiple files (deletion)
- **Test**: Run local benchmark without errors
- **Estimated Time**: 1-2 hours

### Task 1.2: Simplify GitHub Actions Workflows
- **Objective**: Modify workflows to only publish pre-built files
- **Scope**: Update deploy-pages.yml to remove benchmark execution
- **Success Criteria**:
  - [ ] Remove benchmark execution from `deploy-pages.yml`
  - [ ] Keep only static file publishing functionality
  - [ ] Ensure PR preview still works with lightweight benchmark
  - [ ] Remove unused workflow parameters (timeout, etc.)
- **Files to Modify**: `.github/workflows/deploy-pages.yml`
- **Test**: Successful deployment of pre-built docs/
- **Estimated Time**: 1 hour

### Task 1.3: Update Configuration for Local-Only
- **Objective**: Simplify configuration to focus on local execution
- **Scope**: Remove external storage and complex features from config
- **Success Criteria**:
  - [ ] Update `config/benchmark_config.yaml` to remove external storage
  - [ ] Remove `problems/medium_set/` and `problems/large_set/`
  - [ ] Keep only `problems/light_set/` for local execution
  - [ ] Update `config/solvers.yaml` to remove Octave references
- **Files to Modify**: Configuration files in `config/`
- **Test**: Load configuration without errors
- **Estimated Time**: 30 minutes


---

## Phase 2: Code Cleanup (High Priority)

### Task 2.1: Organize Test Files
- **Objective**: Move scattered test files to proper test directory
- **Scope**: Clean up root directory by moving test files
- **Success Criteria**:
  - [ ] Create `tests/unit/`, `tests/integration/`, `tests/fixtures/`
  - [ ] Move `test_*.py` files from root to appropriate test directories
  - [ ] Move `debug_*.py` files to `tests/debug/` or delete if obsolete
  - [ ] Update test runner configurations
- **Files to Modify**: Root directory test files, test configuration
- **Test**: All tests run from new locations
- **Estimated Time**: 1 hour

### Task 2.2: Remove Unused Code and Dependencies
- **Objective**: Clean up unused modules and reduce complexity
- **Scope**: Audit codebase for unused components
- **Success Criteria**:
  - [ ] Remove unused imports across all Python files
  - [ ] Delete unused utility functions
  - [ ] Clean up `requirements/` files to remove unnecessary dependencies
  - [ ] Remove unused configuration templates
- **Files to Modify**: Multiple Python files, requirements files
- **Test**: System functionality unchanged after cleanup
- **Estimated Time**: 2 hours

### Task 2.3: Simplify Main Interface
- **Objective**: Ensure main.py works reliably for local execution
- **Scope**: Verify and simplify main.py command line interface
- **Success Criteria**:
  - [ ] Verify all command line arguments work correctly
  - [ ] Remove any unused or problematic arguments
  - [ ] Ensure data publishing integration works properly
  - [ ] Add clear error messages for common issues
- **Files to Modify**: `main.py`
- **Test**: All main.py operations work without errors
- **Estimated Time**: 1 hour

---

## Phase 3: Documentation Updates (Medium Priority)

### Task 3.1: Update Architecture Documentation
- **Objective**: Update design documents to reflect simplified architecture
- **Scope**: Revise all design documents for new simplified approach
- **Success Criteria**:
  - [ ] Update `docs/development/detail_design.md` with simplified architecture
  - [ ] Update `README.md` with simplified setup instructions
  - [ ] Remove outdated guides about Octave integration
- **Files to Modify**: Architecture and design documentation
- **Test**: Documentation accurately reflects current system
- **Estimated Time**: 2 hours

### Task 3.2: Create Local Development Guide
- **Objective**: Document the new local-first workflow
- **Scope**: Create clear instructions for local benchmark execution
- **Success Criteria**:
  - [ ] Document local benchmark execution process
  - [ ] Explain how to generate and commit reports
  - [ ] Document GitHub Pages publishing workflow
  - [ ] Create troubleshooting guide for common issues
- **Files to Modify**: `docs/guides/LOCAL_DEVELOPMENT.md` (new)
- **Test**: Follow guide to successfully run local benchmarks
- **Estimated Time**: 1-2 hours

---

## Phase 4: Verification and Testing (Medium Priority)

### Task 4.1: Test Complete Local Workflow
- **Objective**: Verify entire local workflow works end-to-end
- **Scope**: Test benchmark execution, report generation, and publishing
- **Success Criteria**:
  - [ ] Run `python main.py --all` successfully
  - [ ] Generate clean HTML reports with proper navigation
  - [ ] Commit generated files to repository
  - [ ] Verify GitHub Actions deploys static files correctly
- **Files to Modify**: None (testing only)
- **Test**: Complete workflow from benchmark to published site
- **Estimated Time**: 1 hour

### Task 4.2: Test PR Preview Functionality
- **Objective**: Ensure PR preview still works with simplified system
- **Scope**: Test PR preview workflow with lightweight benchmark
- **Success Criteria**:
  - [ ] Create test PR to trigger preview workflow
  - [ ] Verify preview deploys to correct subdirectory
  - [ ] Confirm preview shows different data from main site
  - [ ] Test preview cleanup when PR is closed
- **Files to Modify**: None (testing only)
- **Test**: PR preview workflow completes successfully
- **Estimated Time**: 30 minutes

### Task 4.3: Validate Published Site
- **Objective**: Ensure published GitHub Pages site works correctly
- **Scope**: Test all aspects of published site functionality
- **Success Criteria**:
  - [ ] All HTML pages load without errors
  - [ ] Navigation links work correctly
  - [ ] Data files are accessible
  - [ ] Responsive design works on mobile
  - [ ] No broken images or CSS
- **Files to Modify**: None (testing only)
- **Test**: Published site fully functional
- **Estimated Time**: 30 minutes

---

## Phase 5: Final Polish (Low Priority)

### Task 5.1: Update README and Documentation
- **Objective**: Ensure all documentation reflects simplified system
- **Scope**: Final documentation review and updates
- **Success Criteria**:
  - [ ] Update main README with simplified setup instructions
  - [ ] Remove references to removed features
  - [ ] Add badges and links to published site
  - [ ] Update license and contribution information
- **Files to Modify**: `README.md`, various documentation files
- **Test**: Documentation is clear and accurate
- **Estimated Time**: 1 hour

### Task 5.2: Clean Up Repository Structure
- **Objective**: Final cleanup of repository organization
- **Scope**: Ensure clean, professional repository structure
- **Success Criteria**:
  - [ ] Remove any remaining temporary files
  - [ ] Update `.gitignore` to exclude local-only files
  - [ ] Ensure consistent file naming conventions
  - [ ] Verify no sensitive information is committed
- **Files to Modify**: Various files, `.gitignore`
- **Test**: Repository looks professional and clean
- **Estimated Time**: 30 minutes

---

## Implementation Order

### Sprint 1 (Day 1): Core Simplification
1. Task 1.1: Remove Octave Support
2. Task 1.2: Simplify GitHub Actions Workflows  
3. Task 1.3: Update Configuration for Local-Only
4. Task 2.3: Simplify Main Interface

### Sprint 2 (Day 2): Cleanup and Testing
1. Task 2.1: Organize Test Files
2. Task 2.2: Remove Unused Code and Dependencies
3. Task 4.1: Test Complete Local Workflow

### Sprint 3 (Day 3): Documentation and Validation
1. Task 3.1: Update Architecture Documentation
2. Task 3.2: Create Local Development Guide
3. Task 4.2: Test PR Preview Functionality
4. Task 4.3: Validate Published Site

### Sprint 4 (Day 4): Final Polish
1. Task 5.1: Update README and Documentation
2. Task 5.2: Clean Up Repository Structure

---

## Success Criteria for Each Task

Each task must meet these criteria:
1. **Minimal Scope**: Focus on single concern, independently testable
2. **Clear Start/End**: Obvious completion criteria
3. **Testable**: Can verify task completion objectively
4. **No Regression**: Existing functionality continues to work
5. **Documentation**: Changes reflected in relevant documentation

---

## Risk Mitigation

### High Risk Areas:
- GitHub Actions workflow changes (test carefully)
- Removing code dependencies (check all imports)
- Data publishing integration (verify JSON generation)

### Mitigation Strategies:
- Test each change incrementally
- Keep backup of working state
- Use feature branches for risky changes
- Verify published site after each major change

---

## Final Goal

A simplified, reliable optimization solver benchmark system that:
- Runs benchmarks locally
- Generates clean HTML reports
- Publishes to GitHub Pages reliably
- Supports PR previews
- Has minimal complexity and failure points
- Is maintainable and well-documented