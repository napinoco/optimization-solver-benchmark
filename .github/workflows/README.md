# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating the optimization solver benchmark system.

## Benchmark Workflow (benchmark.yml)

The main benchmark workflow (`benchmark.yml`) automatically runs optimization solver benchmarks on every push to the main branch and pull requests.

### Features

- **Automatic Execution**: Runs on push/PR to main branch
- **Manual Trigger**: Can be triggered manually with custom parameters
- **Cross-Platform Testing**: Optional testing on Ubuntu, macOS, and Windows
- **Multi-Python Version**: Tests with Python 3.10, 3.11, and 3.12
- **Artifact Upload**: Saves benchmark results and HTML reports
- **Validation Integration**: Includes result validation and error reporting
- **Summary Generation**: Provides detailed summaries in GitHub UI

### Manual Trigger Parameters

When triggering the workflow manually, you can customize:

- **Solvers**: Comma-separated list (e.g., "scipy,cvxpy")
- **Problem Set**: Choose from light_set, medium_set, or large_set
- **Timeout**: Solver timeout in seconds

### Workflow Steps

1. **Environment Setup**: Install Python, create virtual environment
2. **Dependencies**: Install base and Python solver dependencies
3. **Validation**: Verify environment and configuration
4. **Benchmark Execution**: Run solvers on selected problem sets
5. **Report Generation**: Create HTML reports from results
6. **Artifact Upload**: Save database, logs, and reports
7. **Summary Display**: Show results in GitHub Actions UI
8. **Validation Check**: Report any validation errors/warnings

### Artifacts Generated

- **benchmark-results**: SQLite database and log files (30-day retention)
- **benchmark-reports**: Generated HTML reports (30-day retention)
- **Cross-platform results**: Results from different OS/Python combinations (7-day retention)
- **GitHub Pages**: Automatic deployment of HTML reports (accessible via repository GitHub Pages URL)

### Environment Variables

The workflow automatically sets:
- `GITHUB_ACTIONS=true`: Enables CI/CD mode in the benchmark system
- Python version and OS information for reporting

### Usage Examples

**Automatic execution** (on every push/PR):
```bash
git push origin main  # Triggers workflow with default settings
```

**Manual execution** with custom parameters:
1. Go to Actions tab in GitHub repository
2. Select "Optimization Solver Benchmark" workflow
3. Click "Run workflow"
4. Specify custom solvers, problem set, and timeout
5. Click "Run workflow" button

### Performance Considerations

- **Light set**: Quick execution (~1-2 minutes) for CI/CD
- **Medium set**: Moderate execution (~5-10 minutes) for comprehensive testing
- **Large set**: Extended execution (~30+ minutes) for full validation
- **Cross-platform**: Only runs on manual trigger to save CI minutes

### Monitoring and Debugging

The workflow provides extensive logging and monitoring:

- **Step summaries**: Each step shows execution details
- **Validation reports**: Automatic detection of data quality issues
- **Artifact inspection**: Download results for local analysis
- **Error reporting**: Clear indication of failures with context

### Requirements

The workflow expects these files in the repository:
- `requirements/base.txt`: Core dependencies
- `requirements/python.txt`: Python solver dependencies
- `config/benchmark_config.yaml`: Benchmark configuration
- `main.py`: Main entry point script

### Troubleshooting

Common issues and solutions:

1. **Dependency Installation Failure**:
   - Check requirements files for syntax errors
   - Verify package versions are compatible

2. **Benchmark Execution Failure**:
   - Check solver installation and configuration
   - Verify problem files are valid and accessible

3. **Validation Errors**:
   - Review validation error messages in workflow summary
   - Check data quality and solver behavior

4. **Artifact Upload Issues**:
   - Ensure file paths exist before upload
   - Check GitHub repository storage limits

For more information, see the workflow file: `.github/workflows/benchmark.yml`