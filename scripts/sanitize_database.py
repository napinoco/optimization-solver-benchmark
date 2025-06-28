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
    """Sanitize environment info JSON string"""
    if not env_info_str:
        return "{}"
    
    try:
        env_info = json.loads(env_info_str)
    except json.JSONDecodeError:
        return "{}"
    
    if not isinstance(env_info, dict):
        return "{}"
    
    # Create sanitized copy
    sanitized = {}
    
    # CPU info - safe to keep
    if 'cpu' in env_info:
        cpu = env_info['cpu']
        sanitized['cpu'] = {
            'cpu_count': cpu.get('cpu_count'),
            'cpu_count_physical': cpu.get('cpu_count_physical'),
            'processor': cpu.get('processor')
        }
    
    # Memory info - keep totals only
    if 'memory' in env_info:
        memory = env_info['memory']
        sanitized['memory'] = {
            'total_gb': memory.get('total_gb')
        }
    
    # OS info - keep basic system info only
    if 'os' in env_info:
        os_info = env_info['os']
        sanitized['os'] = {
            'architecture': os_info.get('architecture'),
            'machine': os_info.get('machine'),
            'system': os_info.get('system'),
            'platform': os_info.get('platform'),
            'release': os_info.get('release')
        }
    
    # Python info - remove executable path
    if 'python' in env_info:
        python = env_info['python']
        sanitized['python'] = {
            'implementation': python.get('implementation'),
            'version': python.get('version'),
            'version_info': python.get('version_info')
        }
    
    # Git info - filter sensitive parts
    if 'git' in env_info:
        git = env_info['git']
        sanitized['git'] = {
            'available': git.get('available'),
            'commit_hash': git.get('commit_hash')
            # Note: Removing branch name and is_dirty status
        }
    
    # Timezone info - keep basic info only
    if 'timezone' in env_info:
        tz = env_info['timezone']
        sanitized['timezone'] = {
            'timezone_name': tz.get('timezone_name'),
            'utc_offset_hours': tz.get('utc_offset_hours')
        }
    
    # Timestamp - safe to keep
    if 'timestamp' in env_info:
        sanitized['timestamp'] = env_info['timestamp']
    
    return json.dumps(sanitized)


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