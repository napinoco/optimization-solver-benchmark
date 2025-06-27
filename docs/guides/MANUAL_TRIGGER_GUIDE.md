# Manual Workflow Trigger Guide

This guide explains how to manually trigger the deployment workflow.

## Quick Start

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **"Deploy Reports"** workflow
4. Click **"Run workflow"** button
5. Click **"Run workflow"** to confirm

## What the Manual Trigger Does

The manual workflow trigger simply deploys your pre-built reports to GitHub Pages. It does **not** run benchmarks - you need to generate reports locally first.

### Prerequisites

Before triggering the workflow, ensure you have:

1. **Generated reports locally**:
   ```bash
   python main.py --all
   ```

2. **Committed generated files**:
   ```bash
   git add docs/ database/
   git commit -m "Update benchmark reports"
   git push
   ```

## Workflow Behavior

The deployment workflow:
- ✅ **Verifies** that pre-built reports exist in `docs/`
- ✅ **Deploys** static files to GitHub Pages
- ❌ **Does NOT run benchmarks** (purely deployment)

## Manual Benchmark Generation

To customize your benchmark before manual deployment:

### Basic Options
```bash
# Run all libraries
python main.py --all

# Run specific libraries  
python main.py --all --library_names DIMACS
python main.py --all --library_names SDPLIB
python main.py --all --library_names DIMACS,SDPLIB

# Run specific solvers
python main.py --all --solvers cvxpy_clarabel,cvxpy_scs
```

### Advanced Options
```bash
# Verbose output
python main.py --all --verbose

# Validate before running
python main.py --validate

# Benchmark only (no reports)
python main.py --benchmark --library_names DIMACS
```

## Troubleshooting

### Common Issues

**"No pre-built reports found"**
```bash
# Generate reports first
python main.py --all
git add docs/ database/
git commit -m "Add generated reports"
git push
```

**Workflow fails immediately**
- Ensure you have GitHub Actions enabled
- Verify you committed the generated `docs/` directory
- Check that `docs/index.html` exists

**GitHub Pages not updating**
- Verify Pages is enabled: Settings → Pages → GitHub Actions
- Ensure workflow completed successfully
- Check that you pushed the latest changes

## Best Practices

1. **Always generate reports locally first** - The workflow only deploys, doesn't benchmark
2. **Test locally before deployment** - Use `python main.py --validate` 
3. **Commit generated files** - Include both `docs/` and `database/` in commits
4. **Monitor deployment** - Check Actions tab for deployment status

## Support

For deployment issues:
1. Check workflow logs in GitHub Actions tab
2. Verify you have all required files committed
3. Test report generation locally first

**Local validation:**
```bash
python main.py --validate
python main.py --all
ls docs/index.html docs/data/
```