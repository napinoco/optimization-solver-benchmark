#!/usr/bin/env python3
"""
Analytics Runner
================

Comprehensive analytics runner that executes all advanced statistical analysis,
performance profiling, and visualization generation in sequence.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.analytics.statistical_analysis import run_statistical_analysis
from scripts.analytics.performance_profiler import run_performance_profiling
from scripts.analytics.visualization import create_analytics_dashboard
from scripts.utils.logger import get_logger

logger = get_logger("analytics_runner")


def run_complete_analytics(output_dir: str = "docs", verbose: bool = False) -> bool:
    """
    Run complete analytics pipeline.
    
    Args:
        output_dir: Directory to save reports and visualizations
        verbose: Enable verbose logging
        
    Returns:
        True if all analytics completed successfully, False otherwise
    """
    
    success = True
    
    print("🔬 Starting Comprehensive Analytics Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Statistical Analysis
        print("\n📊 Step 1: Running Advanced Statistical Analysis...")
        run_statistical_analysis()
        print("✅ Statistical analysis completed")
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        success = False
    
    try:
        # Step 2: Performance Profiling
        print("\n⚡ Step 2: Running Performance Profiling...")
        run_performance_profiling()
        print("✅ Performance profiling completed")
        
    except Exception as e:
        print(f"❌ Performance profiling failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        success = False
    
    try:
        # Step 3: Visualization Dashboard
        print("\n🎨 Step 3: Creating Analytics Dashboard...")
        create_analytics_dashboard()
        print("✅ Analytics dashboard created")
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        success = False
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All analytics completed successfully!")
        print(f"\n📄 Generated Reports:")
        print(f"  • Statistical Analysis: {output_dir}/statistical_analysis_report.json")
        print(f"  • Performance Profiling: {output_dir}/performance_profiling_report.json")
        print(f"  • Interactive Dashboard: {output_dir}/performance_dashboard.html")
        
        print(f"\n💡 Next Steps:")
        print(f"  • Open {output_dir}/performance_dashboard.html in your browser")
        print(f"  • Review detailed reports for specific insights")
        print(f"  • Use findings to optimize solver selection and configuration")
        
    else:
        print("⚠️  Some analytics steps failed. Check the logs above for details.")
    
    return success


def run_individual_analytics(component: str, verbose: bool = False) -> bool:
    """
    Run individual analytics component.
    
    Args:
        component: Analytics component to run ('stats', 'profiling', 'dashboard')
        verbose: Enable verbose logging
        
    Returns:
        True if component completed successfully, False otherwise
    """
    
    try:
        if component == 'stats':
            print("📊 Running Statistical Analysis...")
            run_statistical_analysis()
            
        elif component == 'profiling':
            print("⚡ Running Performance Profiling...")
            run_performance_profiling()
            
        elif component == 'dashboard':
            print("🎨 Creating Analytics Dashboard...")
            create_analytics_dashboard()
            
        else:
            print(f"❌ Unknown component: {component}")
            return False
        
        print(f"✅ {component.title()} completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {component.title()} failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Advanced Analytics Runner for Optimization Solver Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run complete analytics pipeline
  %(prog)s --component stats         # Run only statistical analysis
  %(prog)s --component profiling     # Run only performance profiling
  %(prog)s --component dashboard     # Create only dashboard
  %(prog)s --verbose                 # Run with verbose logging
  %(prog)s --output-dir reports/     # Save reports to custom directory
        """
    )
    
    parser.add_argument(
        '--component', 
        choices=['stats', 'profiling', 'dashboard'],
        help='Run specific analytics component only'
    )
    
    parser.add_argument(
        '--output-dir',
        default='docs',
        help='Directory to save reports and visualizations (default: docs)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging and error tracebacks'
    )
    
    parser.add_argument(
        '--list-components',
        action='store_true',
        help='List available analytics components'
    )
    
    args = parser.parse_args()
    
    if args.list_components:
        print("Available Analytics Components:")
        print("  📊 stats      - Advanced statistical analysis with hypothesis testing")
        print("  ⚡ profiling  - Performance profiling with scalability analysis") 
        print("  🎨 dashboard  - Interactive HTML dashboard with visualizations")
        return
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.component:
            # Run individual component
            success = run_individual_analytics(args.component, args.verbose)
        else:
            # Run complete pipeline
            success = run_complete_analytics(args.output_dir, args.verbose)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\\n⏹️ Analytics cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()