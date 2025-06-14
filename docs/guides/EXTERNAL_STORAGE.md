# External Problem Storage Guide

The optimization solver benchmark system supports external storage for large problem sets that are too big to store in the git repository. This guide explains how to use the external storage system.

## Overview

External storage allows the benchmark system to:
- **Download problems on-demand** from URLs (GitHub releases, direct links, etc.)
- **Cache problems locally** to avoid repeated downloads
- **Support large problem sets** without bloating the repository
- **Verify integrity** with checksums and compression
- **Manage storage limits** with automatic cleanup

## Problem Sets

### Light Set (Local)
- **Storage**: Local files in `problems/light_set/`
- **Size**: Small toy problems for development and testing
- **Variables**: < 100
- **Constraints**: < 100
- **File sizes**: < 1KB each

### Medium Set (External)
- **Storage**: External URLs with local caching
- **Registry**: `problems/medium_set/external_urls.yaml`
- **Size**: Medium-sized benchmark problems
- **Variables**: 100 - 10,000
- **Constraints**: 100 - 10,000
- **File sizes**: 1MB - 100MB

### Large Set (External)
- **Storage**: External URLs with local caching
- **Registry**: `problems/large_set/external_urls.yaml`
- **Size**: Large-scale problems for stress testing
- **Variables**: 10,000+
- **Constraints**: 10,000+
- **File sizes**: 100MB - 10GB

## Cache Management

### Cache Location
- **Default**: `~/.optimization_benchmark_cache/`
- **Configurable** via `ExternalProblemStorage(cache_dir="/path/to/cache")`

### Cache Settings
- **Max size**: 5GB for medium set, 10GB for large set
- **TTL**: 1 week for medium, 2 weeks for large problems
- **Cleanup**: Automatic when cache exceeds 80% (medium) or 70% (large) capacity

### Cache Structure
```
~/.optimization_benchmark_cache/
├── cache_metadata.json          # Cache index and metadata
├── netlib_afiro_a1b2c3d4        # Cached problem files
├── maros_aug2d_e5f6g7h8         # (named: problem_url-hash)
└── ...
```

## External Manager CLI

The `external_manager.py` script provides command-line tools for managing external problems.

### List Available Problems
```bash
# List all external problems
python scripts/storage/external_manager.py list

# List problems from specific set
python scripts/storage/external_manager.py list --set medium_set
python scripts/storage/external_manager.py list --set large_set
```

### Download Problems
```bash
# Download all problems from all sets
python scripts/storage/external_manager.py download

# Download all problems from specific set
python scripts/storage/external_manager.py download --set medium_set

# Download specific problems
python scripts/storage/external_manager.py download netlib_afiro maros_aug2d

# Force re-download (ignore cache)
python scripts/storage/external_manager.py download --force netlib_afiro
```

### Cache Management
```bash
# Show cache information
python scripts/storage/external_manager.py cache-info

# Clear all cache
python scripts/storage/external_manager.py clear-cache

# Clear specific problems from cache
python scripts/storage/external_manager.py clear-cache netlib_afiro maros_aug2d
```

### URL Validation
```bash
# Validate all external URLs
python scripts/storage/external_manager.py validate

# Validate specific problem set
python scripts/storage/external_manager.py validate --set medium_set
```

## Registry Format

External problem registries are YAML files that define problem metadata and download URLs.

### Registry Structure
```yaml
version: "1.0"
description: "Problem set description"

problems:
  - name: "problem_name"              # Unique identifier
    url: "https://example.com/file"   # Download URL
    size: 1048576                     # File size in bytes (optional)
    checksum: "sha256_hash"           # SHA256 checksum (optional)
    checksum_type: "sha256"           # Checksum algorithm
    compressed: true                  # Is file compressed?
    compression_type: "gzip"          # Compression format
    cache_ttl_hours: 168              # Cache TTL (1 week)
    metadata:                         # Problem metadata
      problem_type: "LP"              # LP, QP, SOCP, SDP
      source: "NETLIB"                # Data source
      n_variables: 1000               # Number of variables
      n_constraints: 500              # Number of constraints
      description: "Problem description"

cache_settings:
  max_total_size_gb: 5.0             # Maximum cache size
  default_ttl_hours: 168             # Default TTL
  cleanup_threshold: 0.8             # Cleanup when 80% full

download_settings:
  timeout_seconds: 300               # Download timeout
  retry_attempts: 3                  # Retry failed downloads
  chunk_size_kb: 8                   # Download chunk size
  verify_checksums: true             # Verify file integrity
```

### Supported Problem Sources

#### NETLIB (Linear Programming)
- **URL**: GitHub repositories with NETLIB problems
- **Format**: MPS files
- **Problems**: Classic LP test problems (AFIRO, KB2, DEGEN3, PDS-20, etc.)

#### Maros-Mészáros (Quadratic Programming)
- **URL**: GitHub repositories with QP problems
- **Format**: MATLAB .mat files
- **Problems**: Financial optimization, contact mechanics (AUG2D, BOYD1, CONT-300, HUBER)

#### CUTEst (Optimization Test Problems)
- **URL**: CUTEst problem collection
- **Format**: SIF files
- **Problems**: Diverse optimization problems (AVGASA, etc.)

#### CVXPY Examples (SOCP)
- **URL**: CVXPY repository examples
- **Format**: JSON data files
- **Problems**: Portfolio optimization, antenna array design

#### Academic Research Problems
- **URL**: Various research repositories
- **Format**: MPS, SDPA, custom formats
- **Problems**: Railway scheduling, sensor networks, graph problems

## Integration with Benchmark System

### Automatic Loading
When running benchmarks with external problem sets:

```python
from scripts.benchmark.problem_loader import load_problem

# Automatically downloads and caches if needed
problem = load_problem("netlib_afiro", problem_set="medium_set")
problem = load_problem("maros_cont300", problem_set="large_set")
```

### Benchmark Execution
```bash
# Run benchmark on medium set (downloads problems as needed)
python scripts/benchmark/benchmark_runner.py --problem-set medium_set

# Run benchmark on specific external problems
python scripts/benchmark/benchmark_runner.py --problems netlib_afiro,maros_aug2d
```

### Configuration Integration
The external storage system integrates with the main benchmark configuration:

```yaml
# In benchmark config
problem_sets:
  light_set:
    storage: "local"
    max_problems: 10
  medium_set:
    storage: "external"
    cache_size_gb: 5.0
    max_problems: 50
  large_set:
    storage: "external"
    cache_size_gb: 10.0
    max_problems: 20
```

## Error Handling

### Common Issues and Solutions

#### Download Failures
- **Issue**: Network timeout or URL not accessible
- **Solution**: Check internet connection, validate URLs with `validate` command
- **Retry**: Use `--force` flag to retry downloads

#### Cache Full
- **Issue**: Cache exceeds maximum size limit
- **Solution**: Automatic cleanup removes oldest entries
- **Manual**: Use `clear-cache` command to free space

#### Checksum Mismatch
- **Issue**: Downloaded file doesn't match expected checksum
- **Solution**: File is automatically deleted and download retried
- **Update**: Registry may need checksum updates

#### Missing Dependencies
- **Issue**: Required libraries not installed (PyYAML, etc.)
- **Solution**: Install requirements: `pip install -r requirements/python.txt`

### Logging and Diagnostics
The external storage system provides detailed logging:

```python
# Enable debug logging
import logging
logging.getLogger("external_storage").setLevel(logging.DEBUG)

# Check storage diagnostics
from scripts.storage.external_storage import ExternalProblemStorage
storage = ExternalProblemStorage()
info = storage.get_cache_info()
print(f"Cache utilization: {info['utilization_percent']:.1f}%")
```

## Best Practices

### Registry Maintenance
1. **Verify URLs** regularly with the validate command
2. **Update checksums** when files change upstream
3. **Test downloads** before committing registry changes
4. **Document problems** with clear descriptions and metadata

### Cache Management
1. **Monitor cache size** with `cache-info` command
2. **Clear unused problems** periodically
3. **Adjust TTL** based on problem update frequency
4. **Use appropriate cache size** for your use case

### Performance Optimization
1. **Pre-download** frequently used problems
2. **Use compressed formats** when available
3. **Choose appropriate chunk sizes** for your network
4. **Enable parallel downloads** for large problem sets

### Security Considerations
1. **Verify checksums** for integrity and security
2. **Use HTTPS URLs** when possible
3. **Validate file formats** before loading
4. **Monitor cache directory** permissions

This external storage system enables the benchmark platform to scale to thousands of optimization problems while maintaining fast performance and minimal repository size.