#!/bin/bash
# MATLAB License Setup Helper for Docker Environment
# Helps configure MATLAB Individual/Home license for container usage

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
MATLAB License Setup Helper for Docker Environment

Usage: $0 [OPTIONS]

OPTIONS:
    --check         Check current MATLAB license configuration
    --setup         Interactive setup of MATLAB license
    --validate      Validate MATLAB license for container use
    --clean         Clean up license-related temporary files
    --help          Show this help message

EXAMPLES:
    $0 --check                 # Check current license status
    $0 --setup                 # Interactive license setup
    $0 --validate              # Validate license configuration

NOTE:
    This script helps configure MATLAB Individual/Home licenses for
    container usage. It does NOT provide licenses - you must have a
    valid MATLAB license from MathWorks.

LICENSE REQUIREMENTS:
    - MATLAB Individual License (Home License)
    - Valid license.lic file from MathWorks
    - Properly activated MATLAB installation on host

EOF
}

# Check if MATLAB is available on host
check_matlab_host() {
    log "Checking MATLAB availability on host system..."
    
    if command -v matlab >/dev/null 2>&1; then
        MATLAB_VERSION=$(matlab -batch "disp(version); exit" 2>/dev/null | grep -E "^[0-9]+\.[0-9]+" | head -n1 || echo "unknown")
        log_success "MATLAB found: Version $MATLAB_VERSION"
        return 0
    else
        log_warning "MATLAB not found in PATH"
        return 1
    fi
}

# Check MATLAB license directory structure
check_license_structure() {
    log "Checking MATLAB license directory structure..."
    
    MATLAB_DIR="$HOME/.matlab"
    
    if [ ! -d "$MATLAB_DIR" ]; then
        log_error "MATLAB directory not found: $MATLAB_DIR"
        log "Please run MATLAB at least once to create the license directory"
        return 1
    fi
    
    log "✓ MATLAB directory exists: $MATLAB_DIR"
    
    # Check for license directories
    local license_found=false
    for license_dir in "$MATLAB_DIR"/R20*_licenses; do
        if [ -d "$license_dir" ]; then
            log "✓ License directory found: $license_dir"
            license_found=true
            
            # Check for license.lic file
            if [ -f "$license_dir/license.lic" ]; then
                log "✓ License file found: $license_dir/license.lic"
            else
                log_warning "License file not found: $license_dir/license.lic"
            fi
        fi
    done
    
    if [ "$license_found" = false ]; then
        log_warning "No license directories found in $MATLAB_DIR"
        log "Please activate MATLAB license first"
        return 1
    fi
    
    return 0
}

# Check for activation data
check_activation_data() {
    log "Checking MATLAB activation data..."
    
    MATLAB_DIR="$HOME/.matlab"
    
    # Check for various activation files
    local activation_files=(
        "$MATLAB_DIR/MathWorks/MATLAB"
        "$MATLAB_DIR/.matlab_license_token"
    )
    
    for activation_file in "${activation_files[@]}"; do
        if [ -e "$activation_file" ]; then
            log "✓ Activation data found: $activation_file"
        else
            log_warning "Activation data not found: $activation_file"
        fi
    done
}

# Validate license for Docker usage
validate_docker_license() {
    log "Validating MATLAB license for Docker container usage..."
    
    MATLAB_DIR="$HOME/.matlab"
    
    # Check directory permissions
    if [ ! -r "$MATLAB_DIR" ]; then
        log_error "Cannot read MATLAB directory: $MATLAB_DIR"
        return 1
    fi
    
    log "✓ MATLAB directory is readable"
    
    # Check for recent license files
    local recent_license_found=false
    for license_dir in "$MATLAB_DIR"/R20*_licenses; do
        if [ -d "$license_dir" ] && [ "$license_dir" -nt "$(date -d '1 year ago' '+%Y-%m-%d')" 2>/dev/null ]; then
            recent_license_found=true
            log "✓ Recent license directory: $license_dir"
        fi
    done
    
    if [ "$recent_license_found" = false ]; then
        log_warning "No recent license directories found"
        log "Please ensure MATLAB license is properly activated"
    fi
    
    # Check license file validity (basic)
    for license_dir in "$MATLAB_DIR"/R20*_licenses; do
        if [ -f "$license_dir/license.lic" ]; then
            if grep -q "MATLAB" "$license_dir/license.lic" 2>/dev/null; then
                log "✓ License file appears valid"
            else
                log_warning "License file may be invalid or corrupted"
            fi
        fi
    done
}

# Interactive setup
interactive_setup() {
    log "Starting interactive MATLAB license setup..."
    
    echo
    echo "This script will help you set up MATLAB licensing for Docker containers."
    echo
    echo "REQUIREMENTS:"
    echo "1. Valid MATLAB Individual/Home License from MathWorks"
    echo "2. MATLAB installed and activated on this host system"
    echo "3. Internet connection for license validation"
    echo
    
    read -p "Do you have a valid MATLAB license and installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Please obtain and install MATLAB first:"
        log "1. Purchase MATLAB Individual License from MathWorks"
        log "2. Download and install MATLAB"
        log "3. Activate your license"
        log "4. Run this script again"
        exit 0
    fi
    
    # Check if MATLAB is available
    if ! check_matlab_host; then
        log_error "MATLAB installation not detected"
        log "Please ensure MATLAB is installed and available in PATH"
        exit 1
    fi
    
    # Check license structure
    if ! check_license_structure; then
        log_error "MATLAB license not properly configured"
        log "Please run MATLAB and complete license activation"
        exit 1
    fi
    
    # Validate for Docker usage
    validate_docker_license
    
    log_success "MATLAB license setup validation completed!"
    
    echo
    echo "NEXT STEPS:"
    echo "1. Your MATLAB license is ready for Docker container usage"
    echo "2. The license directory ~/.matlab will be mounted in containers"
    echo "3. Run 'docker-compose up' or use docker-run.sh to start containers"
    echo "4. Containers will automatically detect and use your MATLAB license"
    echo
}

# Clean up temporary files
clean_temp_files() {
    log "Cleaning up MATLAB license temporary files..."
    
    # Remove any temporary license files
    find "$HOME/.matlab" -name "*.tmp" -type f -delete 2>/dev/null || true
    find "$HOME/.matlab" -name "*.temp" -type f -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Create license summary
show_license_status() {
    log "MATLAB License Status Summary:"
    echo
    
    # Host MATLAB
    if check_matlab_host >/dev/null 2>&1; then
        echo "✓ Host MATLAB: Available"
    else
        echo "✗ Host MATLAB: Not found"
    fi
    
    # License directory
    if [ -d "$HOME/.matlab" ]; then
        echo "✓ License Directory: $HOME/.matlab"
        
        # Count license directories
        local license_count=$(find "$HOME/.matlab" -name "R20*_licenses" -type d | wc -l)
        echo "✓ License Versions: $license_count found"
        
        # Check for license files
        local license_file_count=$(find "$HOME/.matlab" -name "license.lic" -type f | wc -l)
        if [ "$license_file_count" -gt 0 ]; then
            echo "✓ License Files: $license_file_count found"
        else
            echo "✗ License Files: None found"
        fi
        
    else
        echo "✗ License Directory: Not found"
    fi
    
    echo
    
    # Docker readiness
    if check_license_structure >/dev/null 2>&1 && validate_docker_license >/dev/null 2>&1; then
        echo "✓ Docker Ready: Yes - license can be used in containers"
    else
        echo "✗ Docker Ready: No - license setup incomplete"
    fi
    
    echo
}

# Main execution
main() {
    case "${1:-}" in
        --check)
            show_license_status
            ;;
        --setup)
            interactive_setup
            ;;
        --validate)
            if check_matlab_host && check_license_structure && validate_docker_license; then
                log_success "MATLAB license validation passed"
                exit 0
            else
                log_error "MATLAB license validation failed"
                exit 1
            fi
            ;;
        --clean)
            clean_temp_files
            ;;
        --help|"")
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi