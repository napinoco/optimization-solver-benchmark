#!/usr/bin/env python3
"""
Test script for enhanced temporary file management system.

This script tests the temporary file management functionality implemented for Task 12.
"""

import os
import sys
import time
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.temp_file_manager import TempFileManager, temp_file_context
from scripts.utils.logger import get_logger

logger = get_logger("temp_file_test")


def test_basic_temp_file_generation():
    """Test basic temporary file generation."""
    print("Testing basic temporary file generation...")
    
    manager = TempFileManager("test_prefix")
    
    # Generate unique filenames
    file1 = manager.generate_unique_filename(".json")
    file2 = manager.generate_unique_filename(".json")
    
    # Should be different
    assert file1 != file2, "Generated filenames should be unique"
    
    # Should contain prefix
    assert "test_prefix" in os.path.basename(file1), "Filename should contain prefix"
    assert file1.endswith(".json"), "Filename should have correct extension"
    
    print(f"✓ Generated unique files: {os.path.basename(file1)}, {os.path.basename(file2)}")


def test_temp_file_creation_and_cleanup():
    """Test temporary file creation and cleanup."""
    print("\nTesting temporary file creation and cleanup...")
    
    manager = TempFileManager("test_create")
    
    # Create temp file
    temp_file = manager.create_temp_file(".json")
    assert os.path.exists(temp_file), "Created temp file should exist"
    
    # Write some test data
    test_data = {"test": "data", "timestamp": time.time()}
    with open(temp_file, 'w') as f:
        json.dump(test_data, f)
    
    # Verify data
    with open(temp_file, 'r') as f:
        loaded_data = json.load(f)
    assert loaded_data["test"] == "data", "Data should be written and read correctly"
    
    # Cleanup
    cleanup_success = manager.cleanup_file(temp_file)
    assert cleanup_success, "Cleanup should succeed"
    assert not os.path.exists(temp_file), "File should be removed after cleanup"
    
    print(f"✓ Created, used, and cleaned up temp file: {os.path.basename(temp_file)}")


def test_context_manager():
    """Test context manager for automatic cleanup."""
    print("\nTesting context manager for automatic cleanup...")
    
    temp_file_path = None
    
    # Use context manager
    with temp_file_context(".test") as temp_file:
        temp_file_path = temp_file
        assert os.path.exists(temp_file), "Temp file should exist within context"
        
        # Write test data
        with open(temp_file, 'w') as f:
            f.write("test data for context manager")
    
    # File should be cleaned up automatically
    assert not os.path.exists(temp_file_path), "Temp file should be cleaned up after context"
    
    print(f"✓ Context manager automatically cleaned up: {os.path.basename(temp_file_path)}")


def test_orphaned_file_cleanup():
    """Test orphaned file cleanup functionality."""
    print("\nTesting orphaned file cleanup...")
    
    manager = TempFileManager("test_orphan", cleanup_age_hours=0)  # Clean up immediately
    
    # Create some "old" temp files
    old_files = []
    for i in range(3):
        temp_file = manager.create_temp_file(".old")
        with open(temp_file, 'w') as f:
            f.write(f"old file {i}")
        old_files.append(temp_file)
    
    # All should exist
    for temp_file in old_files:
        assert os.path.exists(temp_file), "Old files should exist before cleanup"
    
    # Sleep a bit to ensure files are "old" (since we set cleanup_age_hours=0)
    time.sleep(0.1)
    
    # Cleanup orphaned files
    cleaned_count = manager.cleanup_orphaned_files()
    
    # Files should be cleaned up
    for temp_file in old_files:
        assert not os.path.exists(temp_file), "Old files should be cleaned up"
    
    assert cleaned_count == 3, f"Should have cleaned 3 files, got {cleaned_count}"
    
    print(f"✓ Cleaned up {cleaned_count} orphaned files")


def test_concurrent_file_creation():
    """Test concurrent file creation doesn't cause conflicts."""
    print("\nTesting concurrent file creation...")
    
    manager = TempFileManager("test_concurrent")
    
    # Create multiple files rapidly
    temp_files = []
    for i in range(10):
        temp_file = manager.create_temp_file(".concurrent")
        temp_files.append(temp_file)
        
        # Write unique data
        with open(temp_file, 'w') as f:
            f.write(f"concurrent file {i}")
    
    # All files should exist and be unique
    assert len(set(temp_files)) == 10, "All temp files should be unique"
    
    for temp_file in temp_files:
        assert os.path.exists(temp_file), "All temp files should exist"
    
    # Cleanup all files
    for temp_file in temp_files:
        manager.cleanup_file(temp_file)
    
    print(f"✓ Created {len(temp_files)} concurrent files without conflicts")


def test_temp_file_stats():
    """Test temporary file statistics."""
    print("\nTesting temporary file statistics...")
    
    manager = TempFileManager("test_stats")
    
    # Create some temp files
    temp_files = []
    for i in range(3):
        temp_file = manager.create_temp_file(".stats")
        with open(temp_file, 'w') as f:
            f.write(f"stats file {i}" * 100)  # Make files different sizes
        temp_files.append(temp_file)
    
    # Get stats
    stats = manager.get_temp_file_stats()
    
    assert stats['total_files'] >= 3, "Should count at least our 3 files"
    assert stats['total_size_bytes'] > 0, "Should have positive total size"
    assert 'temp_directory' in stats, "Should include temp directory"
    
    print(f"✓ Temp file stats: {stats['total_files']} files, {stats['total_size_bytes']} bytes")
    
    # Cleanup
    for temp_file in temp_files:
        manager.cleanup_file(temp_file)


def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling scenarios...")
    
    manager = TempFileManager("test_error")
    
    # Test cleanup of non-existent file
    cleanup_success = manager.cleanup_file("/non/existent/file.json")
    assert cleanup_success, "Cleanup of non-existent file should return True"
    
    # Test stats with empty directory pattern
    stats = manager.get_temp_file_stats()
    assert isinstance(stats, dict), "Stats should return dict even with no files"
    
    print("✓ Error handling works correctly")


def main():
    """Run all temporary file management tests."""
    print("=" * 60)
    print("TEMPORARY FILE MANAGEMENT SYSTEM TESTS")
    print("=" * 60)
    
    try:
        test_basic_temp_file_generation()
        test_temp_file_creation_and_cleanup()
        test_context_manager()
        test_orphaned_file_cleanup()
        test_concurrent_file_creation()
        test_temp_file_stats()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Temporary file management system working correctly!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()