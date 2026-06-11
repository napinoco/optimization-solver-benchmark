# Development Conventions and Standards

Coding standards and engineering guidelines for the optimization solver benchmark system.

---

## Git Commit Format

```
[Type] Brief description (50 chars max)

- Specific changes made
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

For reference style, see the existing runners such as `scripts/solvers/matlab/sedumi_runner.m`. Document function parameters and return values with `%` comments.

---

### Python Code Standards

#### General Principles
- **Linting/Formatting**: `ruff check` and `ruff format` must pass (enforced in CI; configuration in `pyproject.toml`)
- **Type Hints**: Use type annotations for all function parameters and return values
- **Docstrings**: Google-style docstrings for all classes and functions
- **Error Handling**: Explicit exception handling with meaningful messages; bare `except:` is forbidden (ruff E722)

For reference style (module docstrings, Google-style docstrings, error handling), see `scripts/solvers/solver_interface.py`.

#### Naming Conventions
- **Classes**: PascalCase (`BenchmarkRunner`, `SolverInterface`)
- **Functions/Methods**: snake_case (`run_benchmark`, `validate_result`)
- **Variables**: snake_case (`solver_name`, `execution_time`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private Members**: Leading underscore (`_internal_method`, `_private_variable`)

#### Import Organization
Imports are grouped (standard library / third-party / local) and sorted automatically by `ruff check --fix` (isort rules).

### Configuration Standards
- YAML files live in `config/` (`site_config.yaml`, `problem_registry.yaml`); use clear hierarchical structure and comment non-obvious values
- Validate configuration changes with `python main.py --validate`

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

### Markdown Standards
- Use hierarchical headers, specify languages for code blocks, and use relative paths for internal links
- Do not duplicate code in documentation: describe the role briefly and reference the source file. Documentation that repeats what the code says goes stale

---

## Architecture Standards

### Modular Design Principles
1. **Single Responsibility**: Each class/module has one clear purpose
2. **Interface Segregation**: Small, focused interfaces over large ones
3. **Dependency Injection**: Use configuration for dependencies
4. **Error Boundaries**: Isolate failures to prevent cascade effects

### Component Integration
New solvers implement the abstract contract in `scripts/solvers/solver_interface.py` (`solve()` returning a `SolverResult`, plus version detection and compatibility checks). See `scripts/solvers/python/cvxpy_runner.py` for a complete implementation.

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

---

## Security Standards

### Input Validation
Always validate external inputs (file existence, supported formats) before processing; see the loaders in `scripts/data_loaders/python/` for the expected pattern.

### Error Handling
- **Never Expose Sensitive Data**: Sanitize error messages
- **Graceful Degradation**: Continue operation despite individual failures
- **Logging Security**: Avoid logging sensitive configuration values
- **Input Sanitization**: Validate all external inputs

---

## Quality Gates

Before merging: tests pass (`pytest tests/`), lint/format checks pass (`ruff check` / `ruff format --check`), and documentation reflects the current implementation.

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