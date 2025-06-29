#!/usr/bin/env python3
"""
Database Sanitization Script
============================

Sanitize existing database records to remove sensitive information.
This script applies the same sanitization logic as the result_processor.py
to all existing records in the database.
"""

import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.result_processor import BenchmarkResult


def sanitize_environment_info(env_info_str: str) -> str:
    """Sanitize environment info JSON string with UTC timezone unification"""
    if not env_info_str:
        return "{}"
    
    try:
        env_info = json.loads(env_info_str)
    except json.JSONDecodeError:
        return "{}"
    
    if not isinstance(env_info, dict):
        return "{}"
    
    # Create sanitized copy with minimal information
    sanitized = {}
    
    # CPU info - keep essential performance info only
    if 'cpu' in env_info:
        cpu = env_info['cpu']
        sanitized['cpu'] = {
            'cpu_count': cpu.get('cpu_count'),
            'cpu_count_physical': cpu.get('cpu_count_physical'),
            'processor': cpu.get('processor'),
            'architecture': cpu.get('architecture') or env_info.get('os', {}).get('architecture')
        }
    
    # Memory info - keep total only (performance relevant)
    if 'memory' in env_info:
        memory = env_info['memory']
        sanitized['memory'] = {
            'total_gb': memory.get('total_gb')
        }
    
    # OS info - keep basic system info only (no version details that could identify specific systems)
    if 'os' in env_info:
        os_info = env_info['os']
        sanitized['os'] = {
            'system': os_info.get('system'),      # Darwin, Linux, Windows
            'machine': os_info.get('machine'),    # arm64, x86_64
            'release': os_info.get('release')     # Keep for compatibility testing
        }
        # Remove: architecture (duplicated), platform (too detailed), version (too specific)
    
    # Python info - keep version only (remove all paths)
    if 'python' in env_info:
        python = env_info['python']
        sanitized['python'] = {
            'implementation': python.get('implementation'),  # CPython, PyPy
            'version': python.get('version'),                # 3.12.2
            'version_info': python.get('version_info')       # 3.12.2
        }
        # Remove: executable (contains user paths)
    
    # Git info - keep commit hash only (remove branch and dirty status)
    if 'git' in env_info:
        git = env_info['git']
        if git.get('available') and git.get('commit_hash'):
            sanitized['git'] = {
                'commit_hash': git.get('commit_hash')
            }
        # Remove: available, branch, is_dirty (privacy/security sensitive)
    
    # Timezone - UTC ONLY (remove all location-specific timezone info)
    # Replace all timezone info with UTC standard
    sanitized['timezone'] = {
        'timezone_name': 'UTC',
        'utc_offset_hours': 0.0
    }
    # Remove all original timezone data to prevent location identification
    
    # Timestamp - convert to UTC if possible, otherwise keep as-is
    if 'timestamp' in env_info:
        # Keep original timestamp for now, but it should be UTC in future
        sanitized['timestamp'] = env_info['timestamp']
    
    return json.dumps(sanitized, separators=(',', ':'))


def sanitize_database(db_path: str):
    """Sanitize all records in the database"""
    print(f"Sanitizing database: {db_path}")
    
    # Connect to database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all records with environment_info
        cursor.execute("SELECT id, environment_info FROM results WHERE environment_info IS NOT NULL")
        records = cursor.fetchall()
        
        print(f"Found {len(records)} records to sanitize")
        
        # Sanitize each record
        sanitized_count = 0
        for record_id, env_info_str in records:
            try:
                sanitized_env = sanitize_environment_info(env_info_str)
                
                # Update the record
                cursor.execute(
                    "UPDATE results SET environment_info = ? WHERE id = ?",
                    (sanitized_env, record_id)
                )
                sanitized_count += 1
                
                if sanitized_count % 10 == 0:
                    print(f"Sanitized {sanitized_count}/{len(records)} records...")
                    
            except Exception as e:
                print(f"Error sanitizing record {record_id}: {e}")
        
        # Commit changes
        conn.commit()
        print(f"✅ Successfully sanitized {sanitized_count} records")


def main():
    """Main function"""
    db_path = project_root / "database" / "results.db"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1
    
    # Create backup
    backup_path = str(db_path) + ".pre_sanitize_backup"
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Sanitize database
    try:
        sanitize_database(str(db_path))
        print("🎉 Database sanitization completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Database sanitization failed: {e}")
        # Restore backup
        shutil.copy2(backup_path, db_path)
        print("Restored original database from backup")
        return 1


if __name__ == "__main__":
    sys.exit(main())