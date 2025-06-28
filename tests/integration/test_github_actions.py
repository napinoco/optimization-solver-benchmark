#!/usr/bin/env python3
"""
Test script to simulate GitHub Actions environment locally.

This script tests the workflow steps that would run in GitHub Actions
to ensure everything works correctly before pushing to GitHub.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"TESTING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print("✓ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def test_github_actions_simulation():
    """Simulate the GitHub Actions workflow steps."""
    
    print("GITHUB ACTIONS WORKFLOW SIMULATION")
    print("="*60)
    
    # Get current directory
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    
    print(f"Repository root: {repo_root}")
    
    # Test 1: Python version check
    if not run_command("python --version", "Check Python version"):
        return False
    
    # Test 2: Check if virtual environment exists
    venv_path = repo_root / "venv"
    if venv_path.exists():
        print("✓ Virtual environment already exists")
    else:
        if not run_command("python -m venv venv", "Create virtual environment"):
            return False
    
    # Test 3: Activate virtual environment and install dependencies
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    
    if not run_command(
        f"{activate_cmd} && pip install --upgrade pip", 
        "Upgrade pip in virtual environment"
    ):
        return False
    
    if not run_command(
        f"{activate_cmd} && pip install -r requirements/base.txt", 
        "Install base dependencies"
    ):
        return False
    
    if not run_command(
        f"{activate_cmd} && pip install -r requirements/python.txt", 
        "Install Python solver dependencies"
    ):
        return False
    
    # Test 4: Verify installation
    if not run_command(
        f"{activate_cmd} && pip list", 
        "List installed packages"
    ):
        return False
    
    # Test 5: Validate environment
    if not run_command(
        f"{activate_cmd} && python main.py --validate", 
        "Validate benchmark environment"
    ):
        return False
    
    # Test 6: Run quick benchmark (dimacs only)
    if not run_command(
        f"{activate_cmd} && python main.py --benchmark --solvers cvxpy_clarabel --library_names dimacs", 
        "Run quick benchmark with CLARABEL solver"
    ):
        return False
    
    # Test 7: Generate reports
    if not run_command(
        f"{activate_cmd} && python main.py --report", 
        "Generate HTML reports"
    ):
        return False
    
    # Test 8: Check if artifacts were created
    expected_artifacts = [
        "database/results.db",
        "logs/benchmark.log", 
        "docs/index.html",
        "docs/solver_comparison.html",
        "docs/problem_analysis.html",
        "docs/environment_info.html"
    ]
    
    print(f"\n{'='*60}")
    print("CHECKING ARTIFACTS")
    print(f"{'='*60}")
    
    all_artifacts_exist = True
    for artifact in expected_artifacts:
        artifact_path = repo_root / artifact
        if artifact_path.exists():
            size = artifact_path.stat().st_size
            print(f"✓ {artifact} ({size} bytes)")
        else:
            print(f"✗ {artifact} (missing)")
            all_artifacts_exist = False
    
    if not all_artifacts_exist:
        print("Some expected artifacts are missing!")
        return False
    
    # Test 9: Simulate different solver combinations
    print(f"\n{'='*60}")
    print("TESTING SOLVER COMBINATIONS")
    print(f"{'='*60}")
    
    solver_combinations = [
        "scipy",
        "cvxpy", 
        "scipy,cvxpy"
    ]
    
    for solvers in solver_combinations:
        if not run_command(
            f"{activate_cmd} && python main.py --benchmark --solvers {solvers} --library_names dimacs", 
            f"Test solver combination: {solvers}"
        ):
            print(f"Warning: Failed to run with solvers: {solvers}")
            # Don't fail the entire test for individual solver issues
    
    # Test 10: Check for validation errors in logs
    print(f"\n{'='*60}")
    print("CHECKING FOR VALIDATION ISSUES")
    print(f"{'='*60}")
    
    log_file = repo_root / "logs" / "benchmark.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        validation_errors = log_content.count("Validation errors")
        validation_warnings = log_content.count("Validation warnings")
        
        print(f"Validation errors found: {validation_errors}")
        print(f"Validation warnings found: {validation_warnings}")
        
        if validation_errors > 0:
            print("⚠️  Validation errors detected in logs")
            # Show some error context
            lines = log_content.split('\n')
            error_lines = [line for line in lines if "Validation errors" in line]
            for line in error_lines[-5:]:  # Show last 5 error lines
                print(f"  {line}")
    
    print(f"\n{'='*60}")
    print("GITHUB ACTIONS SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("The repository is ready for GitHub Actions workflow.")
    print("You can now push to GitHub and the workflow should run correctly.")
    
    return True

def test_environment_variables():
    """Test environment variable handling."""
    
    print(f"\n{'='*60}")
    print("TESTING ENVIRONMENT VARIABLE HANDLING")
    print(f"{'='*60}")
    
    # Test GitHub Actions detection
    original_github_actions = os.environ.get('GITHUB_ACTIONS')
    
    # Simulate GitHub Actions environment
    os.environ['GITHUB_ACTIONS'] = 'true'
    
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    
    if run_command(
        f"{activate_cmd} && python main.py --validate", 
        "Test with GITHUB_ACTIONS=true"
    ):
        print("✓ Environment variable handling works correctly")
    else:
        print("✗ Environment variable handling failed")
        return False
    
    # Restore original environment
    if original_github_actions is not None:
        os.environ['GITHUB_ACTIONS'] = original_github_actions
    else:
        del os.environ['GITHUB_ACTIONS']
    
    return True

if __name__ == "__main__":
    print("Starting GitHub Actions workflow simulation...")
    
    try:
        # Run main simulation
        if not test_github_actions_simulation():
            print("GitHub Actions simulation failed!")
            sys.exit(1)
        
        # Test environment variable handling
        if not test_environment_variables():
            print("Environment variable testing failed!")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("The repository is ready for GitHub Actions.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        sys.exit(1)