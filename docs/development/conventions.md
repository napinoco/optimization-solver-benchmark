# Development Conventions and Standards

Coding standards and engineering guidelines for the optimization solver benchmark system.

---

## Git Commit Format

```
[Type] Brief description (50 chars max)

- Specific changes made

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Commit Types**: Fix, Add, Update, Refactor, Docs, Test

---

## Coding Standards

### MATLAB Code Standards

#### General Principles
- **Function Indentation**: All code within function blocks must be indented
- **Consistent Spacing**: Use 4 spaces for indentation (no tabs)
- **Clear Function Structure**: Separate function signature from body with proper indentation
- **Nested Functions**: Apply consistent indentation for nested function definitions

#### MATLAB Function Structure
```matlab
function result = example_function(param1, param2)
    % Function description and documentation
    % 
    % Args:
    %   param1: Description of parameter
    %   param2: Description of parameter
    %
    % Returns:
    %   result: Description of return value
    
    % All function body code must be indented
    if nargin < 2
        param2 = default_value;
    end
    
    try
        % Implementation logic with proper indentation
        intermediate_value = process_data(param1);
        result = combine_results(intermediate_value, param2);
        
        % Nested function calls maintain indentation
        if result.status == 'success'
            fprintf('Operation completed successfully\n');
        end
        
    catch ME
        % Error handling with consistent indentation
        fprintf('Error in example_function: %s\n', ME.message);
        result = create_error_result(ME);
    end
    
end

function nested_result = helper_function(data)
    % Nested functions also follow indentation rules
    
    nested_result = struct();
    nested_result.processed_data = data * 2;
    nested_result.timestamp = datestr(now);
    
end
```

#### MATLAB Indentation Rules
1. **Function Body**: All code within `function...end` blocks must be indented by 4 spaces
2. **Control Structures**: `if`, `for`, `while`, `try` blocks require additional indentation
3. **Nested Functions**: Each nested function follows the same indentation rules
4. **Comments**: Maintain indentation level consistent with surrounding code
5. **Line Continuation**: Use proper indentation for multi-line statements

#### MATLAB Documentation Standards
- Use `%` for single-line comments with proper indentation
- Document function parameters and return values
- Include usage examples for complex functions
- Maintain consistent commenting style throughout functions

---

### Python Code Standards

#### General Principles
- **Linting/Formatting**: `ruff check` and `ruff format` must pass (enforced in CI; configuration in `pyproject.toml`)
- **Type Hints**: Use type annotations for all function parameters and return values
- **Docstrings**: Google-style docstrings for all classes and functions
- **Error Handling**: Explicit exception handling with meaningful messages; bare `except:` is forbidden (ruff E722)

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
from scripts.solvers.solver_interface import SolverInterface
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

- **pytest** is the testing framework; configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`). Run with `pytest tests/` (CI runs this in `validate.yml`).
- **Layout**: `tests/unit/` holds component tests that generate synthetic fixtures in `tmp_path` (no problem submodules or external solvers required); `tests/integration/` holds problem registry integrity checks. Shared setup is in `tests/conftest.py`.
- **Dev tools**: pytest and ruff are intentionally not in `requirements.txt`; install with `pip install pytest ruff`.
- **Slow tests**: tests that execute real solvers must be marked `@pytest.mark.slow` and are excluded from default CI runs.
- For naming and structure, follow the existing tests under `tests/unit/`.

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
# Validate configuration and environment
python main.py --validate

# Check for missing dependencies
pip install -r requirements.txt
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

*Last Updated: February 2026*