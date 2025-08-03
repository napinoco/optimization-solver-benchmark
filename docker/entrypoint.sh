#!/bin/bash
# Optimization Solver Benchmark System - Container Entry Point
# Handles container initialization, environment setup, and command execution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Container information
log "=== Optimization Solver Benchmark Container ==="
log "Container Environment: ${CONTAINER_ENV:-production}"
log "Hostname: $(hostname)"
log "User: $(whoami) (UID: $(id -u), GID: $(id -g))"
log "Working Directory: $(pwd)"
log "Python Version: $(python --version 2>&1)"

# Check container memory limits
if [ -f "/sys/fs/cgroup/memory/memory.limit_in_bytes" ]; then
    MEMORY_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    if [ "$MEMORY_LIMIT" -lt 9223372036854775807 ]; then
        MEMORY_GB=$((MEMORY_LIMIT / 1024 / 1024 / 1024))
        log "Memory Limit: ${MEMORY_GB}GB"
    else
        log "Memory Limit: Unlimited"
    fi
fi

# Check Docker environment
if [ -f "/.dockerenv" ]; then
    log "Running in Docker container"
else
    log_warning "Docker environment file not found"
fi

# Validate Python environment
log "Validating Python environment..."
if ! python -c "import sys; assert sys.version_info >= (3, 12)" 2>/dev/null; then
    log_error "Python 3.12+ required"
    exit 1
fi

# Check core dependencies
log "Checking core dependencies..."
PYTHON_DEPS=("numpy" "scipy" "cvxpy" "yaml" "psutil")
for dep in "${PYTHON_DEPS[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        log "✓ $dep available"
    else
        log_error "✗ $dep not available"
        exit 1
    fi
done

# Note: Using Octave for MATLAB compatibility (no license restrictions)

# Check Octave availability
log "Checking Octave (MATLAB compatibility) availability..."
if command -v octave >/dev/null 2>&1; then
    log "✓ Octave executable available"
    # Test basic Octave functionality
    if octave --version >/dev/null 2>&1; then
        log "✓ Octave functional (MATLAB compatibility mode)"
    else
        log_warning "Octave version check failed"
    fi
else
    log_warning "Octave executable not found"
    log "MATLAB-compatible solvers will not be available"
fi

# Validate project structure
log "Validating project structure..."
REQUIRED_DIRS=("scripts" "config" "problems" "database" "docs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        log "✓ Directory $dir exists"
    else
        log_error "✗ Directory $dir missing"
        exit 1
    fi
done

# Check configuration files
REQUIRED_CONFIGS=("config/site_config.yaml" "config/problem_registry.yaml")
for config in "${REQUIRED_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        log "✓ Configuration $config exists"
    else
        log_error "✗ Configuration $config missing"
        exit 1
    fi
done

# Create necessary directories with proper permissions
log "Setting up working directories..."
mkdir -p /tmp/solver-temp
chmod 755 /tmp/solver-temp

# Create database directory if it doesn't exist
if [ ! -d "database" ]; then
    log "Creating database directory..."
    mkdir -p database
fi

# Create docs/pages directory if it doesn't exist
if [ ! -d "docs/pages" ]; then
    log "Creating docs/pages directory..."
    mkdir -p docs/pages
fi

# Set environment variables for optimal execution
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Octave environment (MATLAB compatibility)
export DISPLAY=""
export _JAVA_OPTIONS="-Djava.awt.headless=true"
export OCTAVE_EXECUTABLE="octave"

# Temporary files cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    find /tmp -name "solver-*" -type f -mtime +1 -delete 2>/dev/null || true
    log "Container cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT
trap 'log_error "Container interrupted"; cleanup; exit 130' INT TERM

# Final initialization message
log_success "Container initialization completed successfully"
log "Ready to execute: $*"
log "==========================================="

# Execute the main command
if [ $# -eq 0 ]; then
    log_warning "No command specified, starting interactive shell"
    exec /bin/bash
else
    log "Executing command: $*"
    exec "$@"
fi