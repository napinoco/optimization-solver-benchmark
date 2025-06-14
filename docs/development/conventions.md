# Development Conventions and Standards

This document establishes coding standards, development protocols, and engineering guidelines for the optimization solver benchmark system.

---

## Development Workflow

### Task-Based Development Protocol
1. **Sequential Task Execution**: Complete one task at a time following the priority order in [tasks.md](tasks.md)
2. **Implementation Phase**: Follow task-specific requirements and implementation plans
3. **Testing Phase**: Validate implementation using provided test criteria
4. **Review Phase**: Stop and wait for user approval after each task completion
5. **Commit Phase**: Commit changes only after user confirmation
6. **Proceed**: Move to next task once approved

### Task Status Management
- **✅ Completed**: Task finished and tested successfully
- **🚧 In Progress**: Currently being worked on (limit to ONE task at a time)
- **⏳ Pending**: Not yet started, waiting for dependencies
- **❌ Blocked**: Waiting for external dependencies or decisions

---

## Git Commit Protocol

### Commit Message Format
Follow this standardized format for all commits:

```
[Type] Brief description (50 chars max)

- Specific changes made
- Files modified
- Impact on system functionality
- Test results if applicable

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types
- **Complete Task X**: Major task completion with full implementation
- **Fix**: Bug fixes and error corrections
- **Add**: New features or capabilities
- **Update**: Enhancements to existing functionality
- **Refactor**: Code restructuring without functional changes
- **Docs**: Documentation-only changes
- **Test**: Testing-related changes

### Commit Best Practices
1. **Atomic Commits**: Each commit should represent a single, complete change
2. **Descriptive Messages**: Clearly explain what was changed and why
3. **Test Before Commit**: Ensure all functionality works before committing
4. **Reference Tasks**: Include task numbers when applicable
5. **Clean History**: Avoid "WIP" or temporary commits in main branch

---

## Coding Standards

### Python Code Standards

#### General Principles
- **PEP 8 Compliance**: Follow Python's official style guide
- **Type Hints**: Use type annotations for all function parameters and return values
- **Docstrings**: Google-style docstrings for all classes and functions
- **Error Handling**: Explicit exception handling with meaningful messages

#### Code Structure
```python
# File header example
"""
Module description.

This module provides [functionality description].
"""

import standard_library
import third_party_packages
import local_modules

class ExampleClass:
    """Class description following Google docstring format.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
    
    def __init__(self, param1: str, param2: int) -> None:
        """Initialize the class with required parameters."""
        self.param1 = param1
        self.param2 = param2
    
    def example_method(self, input_data: Dict[str, Any]) -> List[str]:
        """Method description with clear purpose.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            List of processed results
            
        Raises:
            ValueError: If input_data is invalid
        """
        try:
            # Implementation with clear error handling
            result = self._process_data(input_data)
            return result
        except KeyError as e:
            raise ValueError(f"Invalid input data: missing key {e}") from e
```

#### Naming Conventions
- **Classes**: PascalCase (`BenchmarkRunner`, `SolverInterface`)
- **Functions/Methods**: snake_case (`run_benchmark`, `validate_result`)
- **Variables**: snake_case (`solver_name`, `execution_time`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private Members**: Leading underscore (`_internal_method`, `_private_variable`)

#### Import Organization
```python
# Standard library imports
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine

# Local application imports
from scripts.benchmark.solver_interface import SolverInterface
from scripts.utils.config_loader import ConfigLoader
```

### Configuration Standards

#### YAML Configuration Format
```yaml
# Use clear hierarchical structure
benchmark:
  timeout: 300                 # Always include units in comments
  parallel_jobs: 1             # Explain reasoning for non-obvious values
  problem_sets:
    light_set: "problems/light_set"
    medium_set: "problems/medium_set"

# Group related configurations
reporting:
  formats: ["html", "json", "csv"]
  include_environment_info: true
  
# Use descriptive names and document purpose
solver_backends:
  cvxpy:
    default: "CLARABEL"        # Default backend for CVXPY
    available: ["CLARABEL", "SCS", "ECOS", "OSQP"]
```

#### Configuration Validation
- **Schema Validation**: Use YAML schema validation for all config files
- **Range Checking**: Validate numerical values are within acceptable ranges
- **Required Fields**: Clearly specify required vs optional configuration
- **Default Values**: Provide sensible defaults for all optional settings

---

## Testing Standards

### Testing Framework
- **pytest**: Primary testing framework for Python code
- **Coverage**: Maintain >90% code coverage for core functionality
- **Integration Tests**: Test complete workflows end-to-end
- **Performance Tests**: Validate benchmark execution performance

### Test Organization
```python
# Test file structure
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_solver_interface.py
│   ├── test_problem_loader.py
│   └── test_result_collector.py
├── integration/             # Integration tests for workflows
│   ├── test_benchmark_workflow.py
│   └── test_reporting_pipeline.py
├── performance/             # Performance and regression tests
│   └── test_execution_time.py
└── fixtures/               # Test data and configurations
    ├── sample_problems/
    └── test_configs/
```

### Test Standards
```python
# Test function naming and structure
def test_solver_interface_with_valid_problem():
    """Test that solver interface handles valid problems correctly."""
    # Arrange
    solver = MockSolver()
    problem = create_test_problem()
    
    # Act
    result = solver.solve(problem)
    
    # Assert
    assert result.status == "optimal"
    assert result.solve_time > 0
    assert result.objective_value is not None

def test_solver_interface_with_invalid_problem():
    """Test that solver interface handles invalid problems gracefully."""
    solver = MockSolver()
    invalid_problem = None
    
    with pytest.raises(ValueError, match="Problem cannot be None"):
        solver.solve(invalid_problem)
```

### Manual Testing Protocol
Each task must include manual testing steps:
1. **Functionality Testing**: Verify core features work as expected
2. **Error Testing**: Test error conditions and edge cases
3. **Integration Testing**: Verify compatibility with existing system
4. **Performance Testing**: Ensure no significant performance regression

---

## Documentation Standards

### Code Documentation
- **Inline Comments**: Explain complex logic and business rules
- **Function Docstrings**: Document all public functions and methods
- **Class Docstrings**: Describe class purpose and usage patterns
- **Module Docstrings**: Explain module purpose and key components

### Documentation Structure
```markdown
# Standard documentation format
## Overview
Brief description of the component/feature

## Usage
Code examples showing how to use the feature

## Configuration
Configuration options and their effects

## Examples
Real-world usage examples

## Troubleshooting
Common issues and solutions
```

### Markdown Standards
- **Headers**: Use hierarchical structure (H1 → H2 → H3)
- **Code Blocks**: Always specify language for syntax highlighting
- **Links**: Use relative paths for internal documentation
- **Lists**: Use consistent bullet points and numbering
- **Tables**: Include headers and align columns properly

---

## Architecture Standards

### Modular Design Principles
1. **Single Responsibility**: Each class/module has one clear purpose
2. **Interface Segregation**: Small, focused interfaces over large ones
3. **Dependency Injection**: Use configuration for dependencies
4. **Error Boundaries**: Isolate failures to prevent cascade effects

### Component Integration
```python
# Standard interface pattern
class SolverInterface(ABC):
    """Abstract base class defining solver contract."""
    
    @abstractmethod
    def solve(self, problem: Problem) -> SolverResult:
        """Solve the given optimization problem."""
        pass
    
    @abstractmethod
    def is_compatible(self, problem: Problem) -> bool:
        """Check if solver can handle the problem type."""
        pass

# Implementation pattern
class ConcreteSolver(SolverInterface):
    """Concrete implementation with clear responsibilities."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration dependency injection."""
        self.config = config
        self._validate_config()
    
    def solve(self, problem: Problem) -> SolverResult:
        """Implementation with error handling and validation."""
        if not self.is_compatible(problem):
            raise ValueError(f"Solver cannot handle {problem.type}")
        
        try:
            return self._execute_solve(problem)
        except Exception as e:
            return SolverResult(status="error", error_message=str(e))
```

### Data Model Standards
- **Immutable Data**: Use dataclasses or named tuples for data transfer
- **Type Safety**: Leverage type hints and runtime validation
- **Serialization**: Support JSON serialization for all data models
- **Validation**: Include data validation in model constructors

---

## Performance Standards

### Benchmark Execution
- **Timeout Handling**: All solver execution must respect timeout limits
- **Resource Management**: Clean up resources after each benchmark
- **Parallel Execution**: Use configured parallel job limits
- **Memory Efficiency**: Avoid memory leaks in long-running operations

### GitHub Actions Optimization
- **Caching**: Cache dependencies and intermediate results
- **Artifact Management**: Efficiently handle build artifacts
- **Resource Awareness**: Respect GitHub Actions resource limits
- **Execution Time**: Target <5 minutes for light problem sets

---

## Security Standards

### Input Validation
```python
# Always validate external inputs
def load_problem(file_path: str) -> Problem:
    """Load problem with comprehensive validation."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Problem file not found: {file_path}")
    
    if not file_path.endswith(('.mps', '.qps', '.py')):
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Additional validation...
```

### Error Handling
- **Never Expose Sensitive Data**: Sanitize error messages
- **Graceful Degradation**: Continue operation despite individual failures
- **Logging Security**: Avoid logging sensitive configuration values
- **Input Sanitization**: Validate all external inputs

---

## Release Management

### Version Control
- **Semantic Versioning**: Use MAJOR.MINOR.PATCH format
- **Release Branches**: Create release branches for final testing
- **Tag Releases**: Tag all releases with version numbers
- **Changelog**: Maintain detailed changelog for each release

### Quality Gates
1. **All Tests Pass**: Complete test suite must pass
2. **Code Coverage**: Maintain minimum coverage requirements
3. **Documentation Updated**: All documentation reflects current state
4. **Performance Validated**: No significant performance regression
5. **Security Review**: Basic security checklist completed

---

## Troubleshooting Guidelines

### Common Development Issues

#### Configuration Problems
```bash
# Validate configuration syntax
python -c "import yaml; yaml.safe_load(open('config/benchmark_config.yaml'))"

# Check for missing dependencies
pip install -r requirements/python.txt
```

#### Testing Issues
```bash
# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Debug failing tests
pytest tests/unit/test_solver.py::test_specific_function -v -s
```

#### GitHub Actions Debugging
- Check workflow logs for specific error messages
- Validate YAML syntax before pushing
- Test locally using act or similar tools
- Review artifact uploads and downloads

### Performance Debugging
- Profile solver execution using Python profiling tools
- Monitor memory usage during benchmark execution
- Check database query performance
- Validate parallel execution efficiency

---

*These conventions ensure consistent, maintainable, and high-quality code across the optimization solver benchmark system. All contributors should follow these standards to maintain project coherence.*

*Last Updated: December 2025*