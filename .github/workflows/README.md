# GitHub Actions Workflows

Two workflows automate validation and publishing for the benchmark system.

## validate.yml — Codebase Validation

Lightweight CI that runs on pushes and pull requests to `main` (no benchmarking):

1. Install dependencies from the single `requirements.txt`, plus dev tools (pytest, ruff)
2. Lint and format check: `ruff check` / `ruff format --check`
3. Run the test suite: `pytest tests/`
4. System validation: `python main.py --validate` (configs, problem registry, solver availability)

The workflow is designed to fail when something is broken — failures are not masked.

## deploy.yml — GitHub Pages Deployment

Publishes pre-built HTML reports from `docs/pages/` (generated locally with `python main.py --report` and committed):

- **Push to `main`**: deploys to the root of the `gh-pages` branch
- **Pull requests**: deploys a preview to `pr-preview/pr-<number>/` and posts the URL as a PR comment
- **PR close**: removes the preview directory

For details, read the workflow files in this directory.
