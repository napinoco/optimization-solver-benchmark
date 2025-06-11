# GitHub Actions Setup Guide

This guide explains how to enable and use the GitHub Actions workflow for automated benchmark execution.

## Quick Start

1. **Push to GitHub**: The workflow will automatically run on every push to the main branch
2. **View Results**: Check the Actions tab to see workflow execution
3. **Download Artifacts**: Get benchmark results and HTML reports from completed runs

## Workflow Badge

Add this badge to your README to show workflow status:

```markdown
[![Benchmark](https://github.com/YOUR_USERNAME/optimization-solver-benchmark/workflows/Optimization%20Solver%20Benchmark/badge.svg)](https://github.com/YOUR_USERNAME/optimization-solver-benchmark/actions)
```

## Manual Execution

To run benchmarks with custom parameters:

1. Go to repository → Actions tab
2. Select "Optimization Solver Benchmark" workflow  
3. Click "Run workflow" button
4. Configure parameters:
   - **Solvers**: `scipy,cvxpy` (default) or any subset
   - **Problem Set**: `light_set` (fast), `medium_set`, or `large_set`
   - **Timeout**: `300` seconds (default) or custom value
5. Click "Run workflow"

## Results and Artifacts

Each workflow run produces:

- **Benchmark Results**: SQLite database with all execution data
- **HTML Reports**: Interactive web pages with charts and analysis
- **Log Files**: Detailed execution logs for debugging
- **GitHub Summary**: Key metrics displayed in the Actions UI

## Repository Settings

### Required Permissions

Ensure your repository has these settings:

- **Actions**: Enabled (Settings → Actions → General)
- **Workflows**: Allow all actions and reusable workflows
- **Artifact Storage**: Sufficient space for database and reports

### GitHub Pages Setup

To enable automatic report publishing:

1. Go to repository **Settings** → **Pages**
2. Under **Source**, select "Deploy from a branch"
3. Choose branch: **gh-pages**
4. Choose folder: **/ (root)**
5. Click **Save**

The workflow will automatically create the `gh-pages` branch and deploy reports after each successful benchmark run.

After setup, reports will be available at: `https://YOUR_USERNAME.github.io/optimization-solver-benchmark/`

### Environment Secrets

No secrets are required for basic operation. The workflow uses:

- Public Python packages from PyPI
- Repository contents (problems, configuration)
- Standard GitHub Actions environment

### Branch Protection

Consider enabling branch protection for main branch:

- **Require status checks**: Include "Optimization Solver Benchmark"
- **Require branches to be up to date**: Ensures latest changes are tested
- **Include administrators**: Apply rules to all users

## Workflow Configuration

### Default Behavior

- **Triggers**: Push to main, pull requests to main
- **Solvers**: SciPy and CVXPY
- **Problems**: Light problem set (fast execution)
- **Platform**: Ubuntu latest with Python 3.12

### Customizing Defaults

Edit `.github/workflows/benchmark.yml` to change:

```yaml
# Change default solvers
default: 'scipy,cvxpy,custom_solver'

# Change default problem set  
default: 'medium_set'

# Change default timeout
default: '600'
```

### Adding Solvers

To include new solvers in the workflow:

1. Add solver dependencies to `requirements/python.txt`
2. Implement solver in `scripts/solvers/`
3. Update solver configuration in `config/solvers.yaml`
4. Test locally with `python main.py --solvers new_solver`

## Performance Optimization

### Execution Time

Typical execution times by problem set:

- **light_set**: 1-2 minutes (recommended for CI/CD)
- **medium_set**: 5-10 minutes (comprehensive testing)
- **large_set**: 30+ minutes (full validation)

### Resource Usage

The workflow uses:

- **CPU**: 2 cores (GitHub Actions standard)
- **Memory**: ~7GB available (sufficient for most problems)
- **Storage**: ~100MB per run (database + reports)
- **Network**: Downloads Python packages (~200MB)

### Cost Optimization

To minimize GitHub Actions minutes:

- Use light problem set for regular CI/CD
- Reserve larger sets for manual testing
- Disable cross-platform testing unless needed
- Set reasonable timeouts to prevent runaway jobs

## Monitoring and Alerting

### GitHub Notifications

Configure notifications for workflow failures:

1. Settings → Notifications → Actions
2. Enable "Failed workflows only"
3. Choose email, web, or mobile notifications

### Status Monitoring

Monitor workflow health:

- **Success Rate**: Check Actions tab for recent runs
- **Execution Time**: Look for performance regressions
- **Artifact Size**: Monitor storage usage trends
- **Validation Warnings**: Review data quality issues

### Custom Monitoring

For advanced monitoring, you can:

- Parse workflow artifacts programmatically
- Set up external monitoring with GitHub API
- Create custom dashboards using benchmark data
- Integrate with existing CI/CD monitoring tools

## Troubleshooting

### Common Issues

**Workflow doesn't trigger automatically**:
- Check if Actions are enabled in repository settings
- Verify workflow file syntax with GitHub Actions validator
- Ensure push is to the main branch

**Dependencies fail to install**:
- Check requirements files for syntax errors
- Verify package versions are available on PyPI
- Look for compatibility issues between packages

**Benchmarks fail with solver errors**:
- Review solver installation in workflow logs
- Check solver configuration in `config/solvers.yaml`
- Verify problem files are valid and accessible

**Validation errors in results**:
- Check validation error messages in workflow summary
- Review solver behavior with problematic data
- Consider adjusting validation thresholds if appropriate

### Debug Mode

To enable verbose logging:

1. Edit workflow file: `.github/workflows/benchmark.yml`
2. Add environment variable: `DEBUG: true`
3. Re-run workflow to see detailed logs

### Local Testing

Test workflow steps locally:

```bash
# Run the test simulation
python test_github_actions.py

# Test specific components
python main.py --validate
python main.py --benchmark --solvers scipy
python main.py --report
```

## Security Considerations

### Code Execution

The workflow executes:
- Python code from the repository
- External packages from PyPI
- User-provided problem files

### Data Handling

- No sensitive data is processed by default
- All results are stored in public artifacts
- Consider private repositories for proprietary problems

### Access Control

- Workflow runs with repository permissions
- No external services or APIs accessed
- No secrets or credentials required

For more details, see `.github/workflows/benchmark.yml` and `.github/README.md`.