# Manual Workflow Trigger Guide

This guide explains how to manually trigger the benchmark workflow with custom parameters.

## Quick Start

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **"Optimization Solver Benchmark"** workflow
4. Click **"Run workflow"** button
5. Configure parameters as needed
6. Click **"Run workflow"**

## Available Parameters

### Solvers
- **Description**: Comma-separated list of solvers to run
- **Default**: `scipy,cvxpy`
- **Examples**:
  - `scipy` (SciPy only)
  - `cvxpy` (CVXPY only)  
  - `scipy,cvxpy` (both solvers)
- **Note**: Make sure solvers are properly configured in `config/solvers.yaml`

### Problem Set
- **Description**: Which problem set to use for benchmarks
- **Options**:
  - `light_set` (default) - Small problems, fast execution (~1-2 minutes)
  - `medium_set` - Moderate problems, longer execution (~5-10 minutes)
  - `large_set` - Large problems, extended execution (~30+ minutes)
- **Note**: Larger sets consume more GitHub Actions minutes

### Timeout
- **Description**: Maximum time per solver per problem (in seconds)
- **Default**: `300` (5 minutes)
- **Range**: 10-3600 seconds
- **Examples**:
  - `60` - Quick timeout for testing
  - `300` - Standard timeout
  - `1800` - Extended timeout for difficult problems

### Skip Cross-Platform Testing
- **Description**: Whether to skip cross-platform compatibility testing
- **Default**: `true` (skip to save CI minutes)
- **Options**:
  - `true` - Run only on Ubuntu with Python 3.12 (recommended)
  - `false` - Run on Ubuntu, macOS, Windows with Python 3.10, 3.11, 3.12

### Verbose Logging
- **Description**: Enable detailed logging for debugging
- **Default**: `false`
- **Options**:
  - `false` - Standard logging
  - `true` - Verbose logging with debug information

## Example Configurations

### Quick Test Run
- **Solvers**: `scipy`
- **Problem Set**: `light_set`
- **Timeout**: `60`
- **Skip Cross-Platform**: `true`
- **Verbose Logging**: `false`

**Use case**: Quick validation that everything works

### Comprehensive Benchmark
- **Solvers**: `scipy,cvxpy`
- **Problem Set**: `medium_set`
- **Timeout**: `600`
- **Skip Cross-Platform**: `false`
- **Verbose Logging**: `false`

**Use case**: Full testing across multiple platforms

### Debugging Run
- **Solvers**: `cvxpy`
- **Problem Set**: `light_set`
- **Timeout**: `300`
- **Skip Cross-Platform**: `true`
- **Verbose Logging**: `true`

**Use case**: Debugging solver issues with detailed logs

### Performance Testing
- **Solvers**: `scipy,cvxpy`
- **Problem Set**: `large_set`
- **Timeout**: `1800`
- **Skip Cross-Platform**: `true`
- **Verbose Logging**: `false`

**Use case**: Testing performance on large problem sets

## Workflow Behavior

### Automatic Triggers
When the workflow runs automatically (push/PR), it uses these defaults:
- **Solvers**: `scipy,cvxpy`
- **Problem Set**: `light_set`
- **Timeout**: `300`
- **Cross-Platform**: Skipped
- **Verbose**: `false`

### Manual Triggers
When you trigger manually, your specified parameters override the defaults.

### Parameter Validation
The workflow validates input parameters:
- **Timeout**: Must be 10-3600 seconds
- **Invalid values**: Workflow fails with clear error message

### GitHub Actions Minutes
Consider CI minute usage:
- **light_set**: ~5-10 minutes
- **medium_set**: ~15-30 minutes  
- **large_set**: ~60+ minutes
- **Cross-platform**: Multiplies time by ~9 (3 OS × 3 Python versions)

## Monitoring Results

After triggering the workflow:

1. **Workflow Summary**: Shows input parameters and execution status
2. **Step Details**: Each step shows progress and any errors
3. **Artifacts**: Download benchmark results and HTML reports
4. **GitHub Pages**: Reports automatically published (if enabled)
5. **Validation**: Any data quality issues are highlighted

## Troubleshooting

### Workflow Fails to Start
- Check repository has GitHub Actions enabled
- Verify you have write permissions to the repository

### Invalid Parameter Error
- Check timeout is between 10-3600 seconds
- Ensure solver names match those in config files

### Solver Not Found
- Verify solver is configured in `config/solvers.yaml`
- Check solver dependencies are in requirements files

### Cross-Platform Failures
- Some solvers may not work on all platforms
- Windows paths may need adjustment
- Check platform-specific step logs

### GitHub Pages Deployment Fails
- Ensure Pages is enabled in repository settings
- Check "GitHub Actions" is selected as source
- Verify workflow has write permissions

## Best Practices

### Development
- Start with `light_set` for quick feedback
- Use verbose logging when debugging
- Skip cross-platform to save time during development

### Production
- Use `medium_set` or `large_set` for comprehensive results
- Enable cross-platform testing before releases
- Monitor GitHub Actions minute usage

### CI/CD Integration
- Automatic triggers use conservative defaults
- Manual triggers allow customization for specific needs
- Use artifacts to preserve results between runs

## Support

For issues with manual triggers:
1. Check this guide for parameter validation
2. Review workflow logs for specific error messages
3. Test locally with `main.py` using same parameters
4. Check repository issues for known problems

Example local test:
```bash
python main.py --benchmark --solvers scipy,cvxpy --problem-set light_set --verbose
```