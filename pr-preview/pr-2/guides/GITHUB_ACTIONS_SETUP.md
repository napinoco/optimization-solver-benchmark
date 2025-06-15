# GitHub Actions Setup Guide

This guide explains the simplified GitHub Actions workflows for static site publishing and PR previews.

## Overview

Our GitHub Actions workflows follow a **local-first approach**:
- **Benchmarks run locally** and generate reports in `docs/`
- **GitHub Actions only publishes** pre-built static files to GitHub Pages
- **No benchmark execution** in CI for reliability and speed

## Quick Start

1. **Run benchmarks locally**: `python main.py --all`
2. **Commit generated files**: Include `docs/` and `database/` in your commits
3. **Push to GitHub**: The workflow will automatically publish your pre-built reports

## Workflows

### 1. Deploy Pages (`deploy-pages.yml`)

**Purpose**: Publishes pre-built reports to GitHub Pages
**Trigger**: Push to main branch
**Action**: Deploys static files from `docs/` directory

```yaml
# Simplified workflow - no benchmark execution
name: Deploy Pages
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify pre-built reports exist
        run: |
          if [ ! -f "docs/index.html" ]; then
            echo "❌ Error: No pre-built index.html found"
            echo "Please run 'python main.py --all' locally and commit the generated files."
            exit 1
          fi
      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v3
        with:
          path: docs
```

### 2. PR Preview (`pr-preview.yml`)

**Purpose**: Creates preview deployments for pull requests
**Trigger**: Pull request opened/updated
**Action**: Lightweight benchmark + preview deployment

## Workflow Badge

Add this badge to your README to show deployment status:

```markdown
[![Deploy Pages](https://github.com/YOUR_USERNAME/optimization-solver-benchmark/workflows/Deploy%20Pages/badge.svg)](https://github.com/YOUR_USERNAME/optimization-solver-benchmark/actions)
```

## Repository Settings

### Required Permissions

Ensure your repository has these settings:

- **Actions**: Enabled (Settings → Actions → General)
- **Workflows**: Allow all actions and reusable workflows
- **Artifact Storage**: Sufficient space for database and reports

### GitHub Pages Setup

**REQUIRED**: Enable GitHub Pages to publish your benchmark reports.

**Setup Steps**:
1. Go to repository **Settings** → **Pages**
2. Under **Source**, select "GitHub Actions" 
3. Click **Save**
4. Commit and push to trigger deployment

Reports will be available at: `https://YOUR_USERNAME.github.io/optimization-solver-benchmark/`

### Environment Secrets

No secrets required. The workflow only:
- Reads committed files from `docs/` directory
- Deploys static HTML, CSS, JS, and JSON files
- Uses standard GitHub Actions permissions

### Branch Protection

Recommended settings for main branch:
- **Require status checks**: Include "Deploy Pages" 
- **Require branches to be up to date**: Ensures latest reports are deployed
- **Include administrators**: Apply rules consistently

## Local Development Workflow

### Complete Workflow

Since benchmarks run locally, your development workflow is:

1. **Make Changes**: Edit code in `scripts/`, `config/`, or `problems/`
2. **Run Benchmarks**: `python main.py --all` 
3. **Review Reports**: Open `docs/index.html` to verify results
4. **Commit Everything**: Include updated `docs/` and `database/` files
5. **Push to GitHub**: GitHub Actions deploys your pre-built reports

### Best Practices

**Always Commit Generated Files**:
```bash
# Run benchmarks and generate reports
python main.py --all

# Commit everything including generated files
git add -A
git commit -m "Update benchmark results and reports"
git push origin main
```

**Verify Reports Before Committing**:
- Check `docs/index.html` loads correctly
- Verify solver comparison shows expected results
- Ensure no test data (TestSolver, test_problem) appears

### Troubleshooting Deployments

**"No pre-built index.html found" Error**:
```bash
# Generate reports locally first
python main.py --all

# Verify files exist
ls docs/index.html docs/data/

# Commit and push
git add docs/ database/
git commit -m "Add generated reports"
git push
```

**Reports Don't Update**:
- Ensure you committed the updated `docs/` directory
- Check that `git status` shows no uncommitted files in `docs/`
- Verify the workflow completed successfully in Actions tab

## Performance and Cost

### GitHub Actions Usage

Our simplified approach minimizes GitHub Actions usage:

- **Deploy Pages**: ~30 seconds per deployment
- **No compute-intensive benchmarks** in CI
- **No external dependencies** to install
- **Minimal storage** for static files only

### Resource Requirements

**Local Development**:
- Run benchmarks on your machine with full CPU/memory
- Generate reports with your preferred Python environment
- Test with any problem set size

**GitHub Actions**:
- Only deploys static files (HTML, CSS, JS, JSON)
- Uses minimal compute resources
- Fast and reliable deployments

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

**Deployment Fails: "No pre-built reports found"**:
- Generate reports locally: `python main.py --all`
- Commit generated files: `git add docs/ database/`
- Push to trigger deployment

**GitHub Pages Not Working**:
- Verify Pages is enabled in Settings → Pages
- Ensure "GitHub Actions" is selected as source
- Check repository is public (or GitHub Pro for private)

**Reports Show Old Data**:
- Run fresh benchmarks: `python main.py --all`
- Verify `docs/` files were updated locally
- Commit and push updated files

**Workflow Permission Errors**:
- Check Actions are enabled: Settings → Actions → General
- Verify workflow has write permissions to deploy
- Ensure Pages deployment permissions are granted

### Local Testing

Test your setup before pushing:

```bash
# Validate environment and run complete workflow
python main.py --validate
python main.py --all

# Verify generated files
ls docs/index.html docs/data/results.json

# Check reports work
open docs/index.html  # macOS
# or
xdg-open docs/index.html  # Linux
```

### Debug Mode

For deployment issues, check the Actions tab:
1. Go to repository → Actions → Deploy Pages
2. Click on failed workflow run
3. Expand "Deploy to GitHub Pages" step
4. Review error messages and logs

## Security Considerations

### Simplified Security Model

Our local-first approach provides enhanced security:

**No Code Execution in CI**:
- GitHub Actions only deploys static files
- No Python code execution in workflows
- No external package installation in CI

**Data Handling**:
- All benchmark data generated locally
- Only static HTML/JSON files published
- No sensitive computation in cloud environment

**Access Control**:
- Minimal workflow permissions required
- Standard GitHub Pages deployment only
- No secrets or credentials needed

---

## Summary

The simplified GitHub Actions setup focuses on:
- **Reliability**: No complex CI benchmark execution
- **Speed**: Fast static file deployments (~30 seconds)
- **Security**: Minimal permissions and no code execution
- **Cost**: Minimal GitHub Actions usage
- **Developer Experience**: Local control over benchmark execution

For local development workflow details, see [LOCAL_DEVELOPMENT_GUIDE.md](LOCAL_DEVELOPMENT_GUIDE.md).

*Last Updated: June 2025 (Simplified Architecture)*