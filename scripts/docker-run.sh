#!/bin/bash
# Docker Container Execution Helper for Optimization Solver Benchmark
# Provides convenient interface for running benchmarks in Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
DEFAULT_MEMORY="16g"
DEFAULT_CPU="4.0"
DEFAULT_IMAGE="solver-benchmark:octave"
DEFAULT_CONTAINER="benchmark-container"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Logging functions
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

# Help function
show_help() {
    cat << EOF
Docker Container Execution Helper for Optimization Solver Benchmark

Usage: $0 [OPTIONS] [-- COMMAND [ARGS...]]

OPTIONS:
    --memory SIZE       Set memory limit (default: $DEFAULT_MEMORY)
    --cpu CORES         Set CPU limit (default: $DEFAULT_CPU)
    --timeout SECONDS   Set benchmark timeout (default: 120)
    --build             Build container before running
    --no-scip           Exclude SCIP solver when building (faster build)
    --clean             Remove existing container before running
    --shell             Start interactive shell instead of command
    --octave            Enable Octave (MATLAB compatibility mode)
    --verbose           Enable verbose output
    --dry-run           Show commands without executing
    --help              Show this help message

COMMANDS:
    If no command is specified after --, the default help command runs.
    Use -- to separate options from the command to run in container.

EXAMPLES:
    $0                                         # Show help
    $0 --build                                 # Build container
    $0 --shell                                 # Interactive shell
    $0 -- python main.py --validate           # Validate system
    $0 -- python main.py --all                # Run all benchmarks
    $0 --memory 16g -- python main.py --all   # Run with 16GB memory
    $0 --timeout 300 -- python main.py --benchmark --library_names DIMACS
    $0 --matlab -- python main.py --benchmark --problems nb

RESOURCE LIMITS:
    Memory can be specified as: 1g, 2048m, 512000k
    CPU can be specified as: 1.0, 2.5, 4.0 (number of cores)

OCTAVE MODE:
    Octave provides MATLAB compatibility without license restrictions.
    Use --octave to enable MATLAB-compatible solvers via Octave.

EOF
}

# Parse command line arguments
parse_arguments() {
    MEMORY="$DEFAULT_MEMORY"
    CPU="$DEFAULT_CPU"
    TIMEOUT="120"
    BUILD_CONTAINER=false
    INCLUDE_SCIP=true
    CLEAN_CONTAINER=false
    INTERACTIVE_SHELL=false
    ENABLE_OCTAVE=false
    VERBOSE=false
    DRY_RUN=false
    COMMAND_ARGS=()
    PARSING_OPTIONS=true
    
    while [[ $# -gt 0 ]]; do
        if [[ "$PARSING_OPTIONS" == "true" && "$1" == "--" ]]; then
            PARSING_OPTIONS=false
            shift
            continue
        fi
        
        if [[ "$PARSING_OPTIONS" == "true" ]]; then
            case $1 in
                --memory)
                    MEMORY="$2"
                    shift 2
                    ;;
                --cpu)
                    CPU="$2"
                    shift 2
                    ;;
                --timeout)
                    TIMEOUT="$2"
                    shift 2
                    ;;
                --build)
                    BUILD_CONTAINER=true
                    shift
                    ;;
                --no-scip)
                    INCLUDE_SCIP=false
                    shift
                    ;;
                --clean)
                    CLEAN_CONTAINER=true
                    shift
                    ;;
                --shell)
                    INTERACTIVE_SHELL=true
                    shift
                    ;;
                --octave)
                    ENABLE_OCTAVE=true
                    shift
                    ;;
                --verbose)
                    VERBOSE=true
                    shift
                    ;;
                --dry-run)
                    DRY_RUN=true
                    shift
                    ;;
                --help)
                    show_help
                    exit 0
                    ;;
                -*)
                    log_error "Unknown option: $1"
                    show_help
                    exit 1
                    ;;
                *)
                    log_error "Unexpected argument: $1 (use -- to separate options from command)"
                    show_help
                    exit 1
                    ;;
            esac
        else
            COMMAND_ARGS+=("$1")
            shift
        fi
    done
    
    # No conflicting options to validate currently
}

# Detect Docker availability
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon not running. Please start Docker."
        exit 1
    fi
    
    log "✓ Docker available"
}

# Check if Octave mode is enabled
check_octave_mode() {
    if [[ "$ENABLE_OCTAVE" == "true" ]]; then
        log "✓ Octave mode enabled (MATLAB compatibility)"
        return 0
    else
        log "Octave mode disabled, MATLAB-compatible solvers will be unavailable"
        return 1
    fi
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    local build_cmd=(
        docker build
        -f "$PROJECT_ROOT/docker/Dockerfile"
        -t "$DEFAULT_IMAGE"
        --build-arg "USER_ID=$(id -u)"
        --build-arg "GROUP_ID=$(id -g)"
        --build-arg "INCLUDE_SCIP=$INCLUDE_SCIP"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        build_cmd+=(--progress=plain)
    fi
    
    build_cmd+=("$PROJECT_ROOT")
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: ${build_cmd[*]}"
    else
        "${build_cmd[@]}"
        log_success "Docker image built successfully"
    fi
}

# Clean existing container
clean_container() {
    log "Cleaning existing container..."
    
    if docker ps -a --format '{{.Names}}' | grep -q "^$DEFAULT_CONTAINER$"; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log "DRY RUN: docker rm -f $DEFAULT_CONTAINER"
        else
            docker rm -f "$DEFAULT_CONTAINER" >/dev/null 2>&1
            log "✓ Container removed"
        fi
    else
        log "✓ No existing container to clean"
    fi
}

# Run Docker container
run_container() {
    log "Starting Docker container..."
    
    # Prepare Docker run command
    local docker_cmd=(
        docker run
        --rm
        --name "$DEFAULT_CONTAINER"
        --memory="$MEMORY"
        --memory-swap="$MEMORY"
        --oom-kill-disable=false
        --cpus="$CPU"
    )
    
    # Add interactive flags for shell mode
    if [[ "$INTERACTIVE_SHELL" == "true" ]]; then
        docker_cmd+=(-it)
    fi
    
    # Volume mounts
    docker_cmd+=(
        -v "$PROJECT_ROOT:/app:rw"
        -v "$PROJECT_ROOT/database:/app/database:rw"
        -v "$PROJECT_ROOT/docs:/app/docs:rw"
        -v "$PROJECT_ROOT/config:/app/config:ro"
        -v "$PROJECT_ROOT/problems:/app/problems:ro"
    )
    
    # Octave mode configuration (no license needed)
    if check_octave_mode; then
        docker_cmd+=(-e "ENABLE_MATLAB_SOLVERS=true")
        docker_cmd+=(-e "MATLAB_SOLVER_ENGINE=octave")
    else
        docker_cmd+=(-e "ENABLE_MATLAB_SOLVERS=false")
    fi
    
    # Environment variables
    docker_cmd+=(
        -e "BENCHMARK_TIMEOUT=$TIMEOUT"
        -e "CONTAINER_ENV=development"
        -e "PYTHONUNBUFFERED=1"
        -e "TZ=$(date +%Z)"
        -e "USER=$(whoami)"
        -e "USERNAME=$(whoami)"
    )
    
    # Security options
    docker_cmd+=(--security-opt no-new-privileges:true)
    
    # Working directory
    docker_cmd+=(-w /app)
    
    # Image name
    docker_cmd+=("$DEFAULT_IMAGE")
    
    # Command to run
    if [[ "$INTERACTIVE_SHELL" == "true" ]]; then
        docker_cmd+=(/bin/bash)
    elif [[ ${#COMMAND_ARGS[@]} -gt 0 ]]; then
        docker_cmd+=("${COMMAND_ARGS[@]}")
    else
        docker_cmd+=(python main.py --help)
    fi
    
    # Execute command
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: ${docker_cmd[*]}"
    else
        if [[ "$VERBOSE" == "true" ]]; then
            log "Executing: ${docker_cmd[*]}"
        fi
        
        "${docker_cmd[@]}"
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            log_success "Container execution completed successfully"
        else
            log_error "Container execution failed with exit code $exit_code"
        fi
        
        return $exit_code
    fi
}

# Main execution
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Show configuration if verbose
    if [[ "$VERBOSE" == "true" ]]; then
        log "Configuration:"
        log "  Memory: $MEMORY"
        log "  CPU: $CPU"
        log "  Timeout: $TIMEOUT"
        log "  Build: $BUILD_CONTAINER"
        log "  Clean: $CLEAN_CONTAINER"
        log "  Shell: $INTERACTIVE_SHELL"
        log "  Command: ${COMMAND_ARGS[*]:-<default>}"
    fi
    
    # Check Docker availability
    check_docker
    
    # Build image if requested
    if [[ "$BUILD_CONTAINER" == "true" ]]; then
        build_image
    fi
    
    # Clean existing container if requested
    if [[ "$CLEAN_CONTAINER" == "true" ]]; then
        clean_container
    fi
    
    # Run container
    run_container
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi