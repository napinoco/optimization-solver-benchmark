#!/usr/bin/env python3
"""
External Problem Manager
========================

Command-line utility for managing external problem storage and caching.
Provides commands to download, list, clear, and validate external problems.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.storage.external_storage import ExternalProblemStorage, load_external_problem_registry
from scripts.utils.logger import get_logger

logger = get_logger("external_manager")


def list_available_problems(problem_set: str = None) -> None:
    """List all available external problems."""
    
    problem_sets = ["medium_set", "large_set"] if problem_set is None else [problem_set]
    
    for pset in problem_sets:
        registry_file = project_root / "problems" / pset / "external_urls.yaml"
        if not registry_file.exists():
            print(f"❌ Registry file not found: {registry_file}")
            continue
        
        print(f"\n📁 {pset.upper()}:")
        print("=" * 50)
        
        try:
            problems = load_external_problem_registry(str(registry_file))
            
            if not problems:
                print("  No problems found")
                continue
            
            # Group by problem type
            by_type = {}
            for name, problem in problems.items():
                ptype = problem.metadata.get('problem_type', 'Unknown')
                if ptype not in by_type:
                    by_type[ptype] = []
                by_type[ptype].append((name, problem))
            
            for ptype, prob_list in sorted(by_type.items()):
                print(f"\n  📊 {ptype} Problems ({len(prob_list)}):")
                for name, problem in sorted(prob_list):
                    size_mb = problem.size / (1024 * 1024) if problem.size > 0 else 0
                    vars_info = ""
                    if 'n_variables' in problem.metadata:
                        vars_info = f" | {problem.metadata['n_variables']} vars"
                    if 'n_constraints' in problem.metadata:
                        vars_info += f", {problem.metadata['n_constraints']} cons"
                    
                    print(f"    • {name:<20} {size_mb:>6.1f} MB{vars_info}")
                    if problem.metadata.get('description'):
                        print(f"      {problem.metadata['description']}")
                        
        except Exception as e:
            print(f"❌ Error loading {pset}: {e}")


def download_problems(problem_names: List[str], problem_set: str = None, force: bool = False) -> None:
    """Download specific problems or all problems from a set."""
    
    storage = ExternalProblemStorage()
    
    if problem_set:
        # Download from specific set
        registry_file = project_root / "problems" / problem_set / "external_urls.yaml"
        if not registry_file.exists():
            print(f"❌ Registry file not found: {registry_file}")
            return
        
        problems = load_external_problem_registry(str(registry_file))
        
        if problem_names:
            # Download specific problems
            target_problems = {name: prob for name, prob in problems.items() if name in problem_names}
            missing = set(problem_names) - set(target_problems.keys())
            if missing:
                print(f"❌ Problems not found in {problem_set}: {', '.join(missing)}")
                return
        else:
            # Download all problems from set
            target_problems = problems
            
    else:
        # Download from all sets
        target_problems = {}
        for pset in ["medium_set", "large_set"]:
            registry_file = project_root / "problems" / pset / "external_urls.yaml"
            if registry_file.exists():
                problems = load_external_problem_registry(str(registry_file))
                if problem_names:
                    # Filter by names
                    target_problems.update({name: prob for name, prob in problems.items() if name in problem_names})
                else:
                    # All problems
                    target_problems.update(problems)
    
    if not target_problems:
        print("❌ No problems to download")
        return
    
    print(f"📥 Downloading {len(target_problems)} problems...")
    
    successful = 0
    failed = 0
    
    for name, problem in target_problems.items():
        print(f"\n📥 Downloading {name}...")
        try:
            local_path = storage.get_problem(problem, force_download=force)
            if local_path:
                size_kb = Path(local_path).stat().st_size / 1024
                print(f"  ✅ Downloaded: {local_path} ({size_kb:.1f} KB)")
                successful += 1
            else:
                print(f"  ❌ Download failed")
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    print(f"\n📊 Download Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")


def show_cache_info() -> None:
    """Show cache information."""
    
    storage = ExternalProblemStorage()
    cache_info = storage.get_cache_info()
    
    print("💾 Cache Information:")
    print("=" * 50)
    print(f"📁 Cache directory: {cache_info['cache_dir']}")
    print(f"📊 Total entries: {cache_info['total_entries']}")
    print(f"💿 Total size: {cache_info['total_size_mb']:.2f} MB")
    print(f"📏 Max size: {cache_info['max_size_mb']:.2f} MB")
    print(f"📈 Utilization: {cache_info['utilization_percent']:.1f}%")
    
    if cache_info['entries']:
        print(f"\n📄 Cache Entries:")
        for entry in cache_info['entries']:
            status = "⚠️ EXPIRED" if entry['is_expired'] else "✅ Valid"
            print(f"  • {entry['name']:<25} {entry['size_kb']:>8.1f} KB  {status}")


def clear_cache(problem_names: List[str] = None) -> None:
    """Clear cache entries."""
    
    storage = ExternalProblemStorage()
    
    if problem_names:
        print(f"🗑️ Clearing cache for specific problems: {', '.join(problem_names)}")
        for name in problem_names:
            storage.clear_cache(name)
            print(f"  ✅ Cleared: {name}")
    else:
        print("🗑️ Clearing all cache entries...")
        storage.clear_cache()
        print("  ✅ All cache entries cleared")


def validate_urls(problem_set: str = None) -> None:
    """Validate external URLs without downloading."""
    
    import urllib.request
    import urllib.error
    
    problem_sets = ["medium_set", "large_set"] if problem_set is None else [problem_set]
    
    total_checked = 0
    total_valid = 0
    total_invalid = 0
    
    for pset in problem_sets:
        registry_file = project_root / "problems" / pset / "external_urls.yaml"
        if not registry_file.exists():
            print(f"❌ Registry file not found: {registry_file}")
            continue
        
        print(f"\n🔍 Validating URLs for {pset}...")
        print("=" * 50)
        
        try:
            problems = load_external_problem_registry(str(registry_file))
            
            for name, problem in problems.items():
                total_checked += 1
                print(f"🔗 Checking {name}: {problem.url}")
                
                try:
                    # Try to open URL and get headers
                    request = urllib.request.Request(problem.url, method='HEAD')
                    with urllib.request.urlopen(request, timeout=10) as response:
                        content_length = response.headers.get('Content-Length', 'Unknown')
                        content_type = response.headers.get('Content-Type', 'Unknown')
                        print(f"  ✅ Valid - Size: {content_length}, Type: {content_type}")
                        total_valid += 1
                        
                except urllib.error.HTTPError as e:
                    print(f"  ❌ HTTP Error {e.code}: {e.reason}")
                    total_invalid += 1
                except urllib.error.URLError as e:
                    print(f"  ❌ URL Error: {e.reason}")
                    total_invalid += 1
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    total_invalid += 1
                    
        except Exception as e:
            print(f"❌ Error loading {pset}: {e}")
    
    print(f"\n📊 Validation Summary:")
    print(f"  🔗 Total URLs checked: {total_checked}")
    print(f"  ✅ Valid: {total_valid}")
    print(f"  ❌ Invalid: {total_invalid}")


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="External Problem Manager for optimization benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                           # List all available problems
  %(prog)s list --set medium_set          # List medium set problems
  %(prog)s download                       # Download all problems
  %(prog)s download --set large_set       # Download all large set problems
  %(prog)s download netlib_afiro maros_aug2d  # Download specific problems
  %(prog)s download --force netlib_afiro  # Force re-download
  %(prog)s cache-info                     # Show cache information
  %(prog)s clear-cache                    # Clear all cache
  %(prog)s clear-cache netlib_afiro       # Clear specific problem cache
  %(prog)s validate                       # Validate all URLs
  %(prog)s validate --set medium_set      # Validate medium set URLs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available external problems')
    list_parser.add_argument('--set', choices=['medium_set', 'large_set'], 
                           help='Specific problem set to list')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download external problems')
    download_parser.add_argument('problems', nargs='*', help='Specific problems to download')
    download_parser.add_argument('--set', choices=['medium_set', 'large_set'], 
                               help='Problem set to download from')
    download_parser.add_argument('--force', action='store_true', 
                               help='Force re-download even if cached')
    
    # Cache info command
    subparsers.add_parser('cache-info', help='Show cache information')
    
    # Clear cache command
    clear_parser = subparsers.add_parser('clear-cache', help='Clear cache entries')
    clear_parser.add_argument('problems', nargs='*', help='Specific problems to clear from cache')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate external URLs')
    validate_parser.add_argument('--set', choices=['medium_set', 'large_set'], 
                                help='Specific problem set to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_available_problems(args.set)
        elif args.command == 'download':
            download_problems(args.problems, args.set, args.force)
        elif args.command == 'cache-info':
            show_cache_info()
        elif args.command == 'clear-cache':
            clear_cache(args.problems if args.problems else None)
        elif args.command == 'validate':
            validate_urls(args.set)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()