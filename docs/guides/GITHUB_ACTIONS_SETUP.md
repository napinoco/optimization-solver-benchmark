# GitHub Actions Setup Guide

How CI and publishing work for this repository, and how to set them up.

## Local-First Approach

- **Benchmarks run locally** and generate reports in `docs/pages/`
- **GitHub Actions only validates the codebase and publishes** pre-built static files to GitHub Pages
- No benchmark execution in CI, for reliability and speed

## Workflows

The repository has two workflows (see [.github/workflows/README.md](../../.github/workflows/README.md) and the workflow files themselves for details):

- **`validate.yml`**: lint/format checks (ruff), test suite (pytest), and system validation (`python main.py --validate`) on pushes and PRs to `main`
- **`deploy.yml`**: deploys the committed `docs/pages/` to GitHub Pages; PRs get a preview at `pr-preview/pr-<number>/` (see [PR_PREVIEW_GUIDE.md](PR_PREVIEW_GUIDE.md))

## Repository Settings

1. **Actions**: enable under Settings → Actions → General
2. **GitHub Pages**: Settings → Pages → deploy from the `gh-pages` branch
3. No secrets are required

Reports will be available at `https://<username>.github.io/optimization-solver-benchmark/`.

## Publishing Workflow

Since benchmarks run locally:

```bash
# 1. Run benchmarks and generate reports
python main.py --all

# 2. Verify locally
open docs/pages/index.html   # macOS (Linux: xdg-open)

# 3. Commit generated files and push
git add docs/pages/
git commit -m "Update benchmark reports"
git push
```

Pushing to `main` triggers the deployment automatically.

## Manual Deployment Trigger

`deploy.yml` supports `workflow_dispatch`. The manual trigger only deploys the committed `docs/pages/` — it does **not** run benchmarks, so generate and commit reports first (see above).

1. Go to the repository → **Actions** tab
2. Select the deploy workflow
3. Click **Run workflow**

## Troubleshooting

**Deployment fails with "No pre-built index.html found"**
Reports were not committed. Run `python main.py --report` (or `--all`) locally, then commit `docs/pages/` and push.

**GitHub Pages not updating**
- Verify Pages is enabled (Settings → Pages) and the workflow run succeeded in the Actions tab
- Confirm `git status` shows no uncommitted files under `docs/pages/`

**Validation workflow fails**
The validation workflow is designed to fail when something is actually broken (lint errors, test failures, missing solvers). Reproduce locally with `ruff check scripts/ main.py tests/`, `pytest tests/`, and `python main.py --validate`.

---

For local development workflow details, see [LOCAL_DEVELOPMENT_GUIDE.md](LOCAL_DEVELOPMENT_GUIDE.md).
