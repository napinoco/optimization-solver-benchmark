# Pull Request Preview Guide

This guide explains how to use the PR preview feature to review benchmark report changes before merging.

---

## Overview

The PR preview system automatically generates temporary preview URLs for every pull request, allowing you to review benchmark report changes before merging to main.

### Key Features

- 🚧 **Automatic Deployment**: Preview is generated for every PR automatically
- 🔄 **Auto-Updates**: Preview updates when you push new commits to the PR
- 💬 **PR Comments**: Automatic comment with preview links
- 🧹 **Auto-Cleanup**: Preview is removed when PR is closed
- 📊 **Full Reports**: Complete benchmark report suite available in preview

---

## How It Works

### For Pull Request Authors

1. **Create/Update PR**: When you open a PR or push commits, the preview workflow starts
2. **Wait for Build**: GitHub Actions runs a lightweight benchmark and generates reports
3. **Review Preview**: Check the automatically posted comment for preview links
4. **Iterate**: Push new commits to update the preview automatically

### For Reviewers

1. **Check PR Comment**: Look for the "📊 Benchmark Preview Ready!" comment
2. **Review Reports**: Click preview links to review the changes
3. **Compare with Main**: Preview URL vs production URL for comparison
4. **Provide Feedback**: Comment on specific changes or improvements

---

## Preview Configuration

### Benchmark Settings

The preview uses a lightweight configuration for fast deployment:

- **Solvers**: `scipy,cvxpy` (core solvers only)
- **Problem Set**: `light_set` (small problems for speed)
- **Timeout**: 300 seconds
- **Platform**: Ubuntu 22.04 only

### Preview Features

- **Preview Banner**: Visual indicator that this is a preview environment
- **PR Information**: Links back to the originating pull request
- **Metadata**: JSON file with PR details for debugging
- **Complete Reports**: All report types available (dashboard, comparison, analysis, etc.)

---

## Preview URLs

### URL Structure
```
https://[owner].github.io/[repo]/pr-preview/pr-[number]/
```

### Example URLs
```
Main Dashboard:    /pr-preview/pr-123/
Solver Comparison: /pr-preview/pr-123/solver_comparison.html
Problem Analysis:  /pr-preview/pr-123/problem_analysis.html
Results Matrix:    /pr-preview/pr-123/results_matrix.html
Statistical Analysis: /pr-preview/pr-123/statistical_analysis.html
Performance Profiling: /pr-preview/pr-123/performance_profiling.html
Environment Info:  /pr-preview/pr-123/environment_info.html
```

---

## Available Reports in Preview

### Core Reports
1. **📊 Dashboard** (`index.html`) - Main benchmark overview with key metrics
2. **⚡ Solver Comparison** (`solver_comparison.html`) - Performance comparison between solvers
3. **📋 Problem Analysis** (`problem_analysis.html`) - Problem-specific insights and statistics

### Advanced Reports
4. **📈 Results Matrix** (`results_matrix.html`) - Problems × Solvers comparison matrix
5. **📊 Statistical Analysis** (`statistical_analysis.html`) - Advanced statistical analysis
6. **⚡ Performance Profiling** (`performance_profiling.html`) - Detailed performance metrics
7. **🖥️ Environment Info** (`environment_info.html`) - System specifications and solver info

### Data Downloads
- **Raw Data** (`data/results.json`) - Complete benchmark results in JSON format
- **CSV Export** (`data/results.csv`) - Results in spreadsheet format
- **Metadata** (`data/metadata.json`) - Environment and configuration information

---

## Workflow Details

### Trigger Events

The preview workflow runs on:
- **PR Opened**: Initial preview deployment
- **PR Reopened**: Re-deploy preview if PR was closed and reopened
- **PR Synchronized**: Update preview when new commits are pushed
- **PR Closed**: Clean up preview deployment

### Deployment Process

1. **Environment Setup**: Python 3.12, dependency installation
2. **Validation**: Environment and configuration validation
3. **Benchmark Execution**: Lightweight benchmark run
4. **Report Generation**: HTML report generation with preview indicators
5. **Preview Deployment**: Deploy to GitHub Pages pr-preview branch
6. **Comment Creation**: Post preview links in PR comment

### Cleanup Process

When a PR is closed:
- Preview files are automatically removed from gh-pages branch
- No manual cleanup required
- Preview URLs become inaccessible

---

## Troubleshooting

### Common Issues

**Preview Not Deploying**
- Check GitHub Actions tab for workflow status
- Ensure repository has GitHub Pages enabled
- Verify GitHub Actions permissions include Pages write access

**Preview Out of Date**
- Preview updates automatically on new commits
- Force update by closing and reopening the PR
- Check workflow logs for any errors

**Preview Links Not Working**
- Ensure GitHub Pages is configured for "GitHub Actions" source
- Check that repository has Pages enabled in Settings
- Verify the pr-preview-action has proper permissions

### Debug Information

**Preview Metadata**
Each preview includes a `preview-info.json` file with:
- PR number and title
- Commit SHA
- Generation timestamp
- Repository information
- PR URL

**Workflow Logs**
Check the "Deploy PR Preview" workflow in GitHub Actions for:
- Build logs
- Deployment status
- Error messages
- Environment information

---

## Configuration

### Customizing Preview Behavior

The preview system can be customized by editing `.github/workflows/pr-preview.yml`:

**Change Benchmark Configuration:**
```yaml
# Modify the benchmark execution step
python main.py --benchmark --solvers "scipy,cvxpy,clarabel" --problem-set "medium_set"
```

**Modify Preview Banner:**
```yaml
# Update the preview notice injection
sed -i '/<header>/a\<div>Custom preview notice</div>' docs/*.html
```

**Adjust Workflow Triggers:**
```yaml
on:
  pull_request:
    types: [opened, reopened, synchronize]  # Remove 'closed' to disable cleanup
```

### Repository Settings

**Enable GitHub Pages:**
1. Go to repository Settings → Pages
2. Select "GitHub Actions" as source
3. Ensure "Build and deployment" is set to "GitHub Actions"

**Required Permissions:**
The workflow needs these permissions:
- `contents: read` - Access repository files
- `pages: write` - Deploy to GitHub Pages
- `id-token: write` - GitHub Pages deployment
- `pull-requests: write` - Comment on PRs

---

## Best Practices

### For Contributors

1. **Test Locally First**: Run `python main.py --validate` before pushing
2. **Review Preview**: Always check the preview before requesting review
3. **Update Documentation**: Include documentation changes in PRs
4. **Small Commits**: Make focused commits for easier preview review

### For Reviewers

1. **Check Both**: Review both code changes and preview output
2. **Compare Outputs**: Compare preview reports with current production
3. **Test Scenarios**: Consider different problem sets and solver configurations
4. **Performance Impact**: Review any performance implications

### For Maintainers

1. **Monitor Workflows**: Keep an eye on preview workflow success rates
2. **Update Dependencies**: Keep pr-preview-action updated
3. **Manage Storage**: Monitor GitHub Pages storage usage
4. **Security Review**: Regularly review workflow permissions

---

## Technical Implementation

### Dependencies

- **rossjrw/pr-preview-action@v1**: Core preview deployment functionality
- **actions/github-script@v7**: PR commenting and interaction
- **Standard GitHub Actions**: checkout, setup-python, upload-artifact

### Architecture

```
PR Creation/Update
       ↓
GitHub Actions Workflow
       ↓
Benchmark Execution (Light)
       ↓
Report Generation + Preview Indicators
       ↓
Deploy to gh-pages (pr-preview/pr-N/)
       ↓
Comment on PR with Links
```

### File Structure

```
gh-pages branch:
├── pr-preview/
│   ├── pr-123/          # Preview for PR #123
│   │   ├── index.html
│   │   ├── solver_comparison.html
│   │   └── ...
│   └── pr-124/          # Preview for PR #124
└── ...                  # Main site files
```

---

## Security Considerations

### Safe Practices

- **No Secrets**: Preview workflows don't expose sensitive information
- **Isolated Environment**: Each preview runs in isolated GitHub Actions environment
- **Limited Access**: Preview URLs are public but temporary
- **No Persistence**: Preview data is cleaned up automatically

### Limitations

- **Public Visibility**: Preview URLs are publicly accessible
- **Resource Usage**: Each preview consumes GitHub Actions minutes
- **Storage Limits**: GitHub Pages has storage and bandwidth limits
- **Fork Limitations**: PR previews from forks have limited functionality

---

*This guide covers the complete PR preview system. For questions or issues, please open a GitHub issue or refer to the workflow logs in GitHub Actions.*