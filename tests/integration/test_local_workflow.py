#!/usr/bin/env python3
"""
Local Workflow Testing Script
============================

Test the same commands that GitHub Actions workflows run locally.
This helps debug issues without pushing to GitHub repeatedly.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        
        if result.stdout:
            print("📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Command succeeded")
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False

def test_pr_preview_workflow():
    """Test the PR preview workflow commands locally."""
    print("🚀 Testing PR Preview Workflow Locally")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # 1. Validate environment
    if not run_command("python main.py --validate", "Validate Environment"):
        return False
    
    # 2. Run lightweight benchmark (same as PR preview)
    if not run_command(
        'python main.py --benchmark --solvers "cvxpy_clarabel,cvxpy_scs" --library_names "dimacs"',
        "Run PR Preview Benchmark"
    ):
        print("⚠️  Benchmark failed, but continuing to test report generation...")
    
    # 3. Generate reports
    if not run_command("python main.py --report", "Generate Reports"):
        return False
    
    # 4. Check what files were generated
    print(f"\n{'='*60}")
    print("📁 Generated HTML Files:")
    print(f"{'='*60}")
    
    docs_dir = Path("docs")
    html_files = list(docs_dir.glob("*.html"))
    
    if html_files:
        for html_file in sorted(html_files):
            size = html_file.stat().st_size
            print(f"  ✅ {html_file.name} ({size:,} bytes)")
    else:
        print("  ❌ No HTML files found!")
        return False
    
    # 5. Check for specific expected files
    expected_files = [
        "index.html",
        "solver_comparison.html", 
        "problem_analysis.html",
        "results_matrix.html",
        "statistical_analysis.html",
        "performance_profiling.html",
        "environment_info.html"
    ]
    
    print(f"\n{'='*60}")
    print("🔍 Expected Files Check:")
    print(f"{'='*60}")
    
    missing_files = []
    for expected_file in expected_files:
        file_path = docs_dir / expected_file
        if file_path.exists():
            print(f"  ✅ {expected_file}")
        else:
            print(f"  ❌ {expected_file} (MISSING)")
            missing_files.append(expected_file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    # 6. Test that main files have content
    print(f"\n{'='*60}")
    print("📄 Content Check:")
    print(f"{'='*60}")
    
    index_path = docs_dir / "index.html"
    if index_path.exists():
        content = index_path.read_text()
        if "results_matrix.html" in content:
            print("  ✅ index.html contains link to results_matrix.html")
        else:
            print("  ⚠️  index.html missing link to results_matrix.html")
            
        if len(content) > 1000:
            print(f"  ✅ index.html has substantial content ({len(content):,} chars)")
        else:
            print(f"  ⚠️  index.html seems too small ({len(content):,} chars)")
    
    print(f"\n{'='*60}")
    print("🎉 Local workflow test completed!")
    print("💡 You can now open docs/index.html in your browser to see the result")
    print(f"{'='*60}")
    
    return True

def test_main_deployment_workflow():
    """Test the main deployment workflow commands locally."""
    print("🚀 Testing Main Deployment Workflow Locally")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # 1. Validate environment
    if not run_command("python main.py --validate", "Validate Environment"):
        return False
    
    # 2. Run full benchmark (same as main deployment)
    if not run_command(
        'python main.py --benchmark --solvers "scipy,clarabel_cvxpy,scs_cvxpy,ecos_cvxpy,osqp_cvxpy" --problem-set "standard_set"',
        "Run Main Deployment Benchmark"
    ):
        print("⚠️  Benchmark failed, but continuing to test report generation...")
    
    # 3. Generate reports
    if not run_command("python main.py --report", "Generate Reports"):
        return False
    
    print(f"\n🎉 Main deployment workflow test completed!")
    print("💡 You can now open docs/index.html in your browser to see the result")
    
    return True

def main():
    """Main test function."""
    print("🧪 Local Workflow Testing Tool")
    print("=" * 60)
    print("This script tests the same commands that run in GitHub Actions")
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Choose which workflow to test:")
        print()
        print("1. PR Preview Workflow (lightweight: cvxpy_clarabel,cvxpy_scs + dimacs)")
        print("2. Main Deployment Workflow (full: scipy,clarabel_cvxpy,scs_cvxpy,ecos_cvxpy,osqp_cvxpy + standard_set)")
        print("3. Both workflows")
        print()
        
        try:
            choice = input("Enter choice (1/2/3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Test interrupted by user")
            return 1
    
    try:
        if choice == "1":
            success = test_pr_preview_workflow()
        elif choice == "2":
            success = test_main_deployment_workflow()
        elif choice == "3":
            success1 = test_pr_preview_workflow()
            success2 = test_main_deployment_workflow()
            success = success1 and success2
        else:
            print("❌ Invalid choice")
            return 1
            
        if success:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())