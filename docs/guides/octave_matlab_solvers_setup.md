# Octave MATLAB Solvers Setup Guide

**Complete setup guide for running SeDuMi and SDPT3 optimization solvers with Octave (MATLAB alternative) in Docker environments**

---

## Overview

This guide documents the complete process of setting up SeDuMi and SDPT3 optimization solvers to work with GNU Octave as a free, open-source alternative to MATLAB. The process involves addressing architecture compatibility, Octave-specific function limitations, and Docker environment configuration.

**Target Environment:**
- Docker container with ARM64 architecture (`aarch64-unknown-linux-gnu`)
- GNU Octave 6.4.0+ as MATLAB compatibility layer
- SeDuMi and SDPT3 solvers via git submodules

**Final Results:**
- ✅ SeDuMi: Solving SOCP problems in ~1s with OPTIMAL status
- ✅ SDPT3: Solving SOCP problems in ~4s with OPTIMAL status  
- ✅ 100% success rate for supported problem types

---

## Problem Categories and Solutions

### 1. 🏗️ Architecture Compatibility Issues

#### **Problem: Unrecognized ARM64 Architecture**
```
Error: Unexpected computer type: aarch64-unknown-linux-gnu
```

**Root Cause:** Both SeDuMi and SDPT3 installation scripts did not recognize the ARM64 Linux architecture used in Docker containers.

#### **Solution: Add ARM64 Support to Installation Scripts**

**File: `scripts/solvers/matlab_octave/sedumi/install_sedumi.m`**
```matlab
% Add this case to the switch statement around line 135
case 'aarch64-unknown-linux-gnu'
    mdir = 'o_lin';  % Linux ARM64, use same directory as Linux x64
```

**File: `scripts/solvers/matlab_octave/sdpt3/install_sdpt3.m`**
```matlab
% Add this case to the switch statement around line 33
case 'aarch64-unknown-linux-gnu'
    mdir = 'o_lin';  % Linux ARM64, use same directory as Linux x64
```

**Verification:**
```bash
# Test compilation in Docker
./scripts/docker-run.sh --octave -- octave --no-gui --eval "setup_matlab_solvers"
```

### 2. 🖥️ GUI and Display Issues

#### **Problem: X11 Display Warnings Causing Process Failures**
```
octave: X11 DISPLAY environment variable not set
octave: disabling GUI features
MATLAB subprocess failed (code 1)
```

**Root Cause:** Octave attempted to initialize GUI features in headless Docker environment.

#### **Solution: Use --no-gui Flag for Octave**

**File: `scripts/solvers/matlab_octave/matlab_process_interface.py`**
```python
# Build command array (around line 240)
if self.use_octave:
    cmd = [self.matlab_executable, '--no-gui', '--eval', matlab_command]
else:
    # Use minimal options, rely on environment variables for Java/X11 control
    cmd = [self.matlab_executable, '-batch', matlab_command]
```

**Also update solver detection method (around line 507):**
```python
if self.use_octave:
    cmd = [self.matlab_executable, '--no-gui', '--eval', 'disp("Octave_OK")']
```

### 3. 🔄 Octave-MATLAB Function Compatibility

#### **Problem: Missing MATLAB Functions in Octave**
```
Error: 'jsonencode' undefined
Error: 'startsWith' undefined  
Error: 'endsWith' undefined
Error: 'contains' undefined
```

**Root Cause:** Octave lacks several MATLAB functions introduced in newer MATLAB versions.

#### **Solution: Implement Octave Compatibility Layer**

**File: `scripts/solvers/matlab_octave/matlab_solver_runner.m`**

Add these compatibility functions at the top of the file:

```matlab
function json_str = octave_jsonencode(data)
    % JSON encoding with Octave fallback
    if exist('jsonencode', 'builtin') || exist('jsonencode', 'file')
        json_str = jsonencode(data);
    else
        json_str = simple_json_encode(data);
    end
end

function json_str = simple_json_encode(data)
    % Basic JSON encoder for Octave compatibility
    if isstruct(data)
        fields = fieldnames(data);
        json_parts = {};
        for i = 1:length(fields)
            field_name = fields{i};
            field_value = data.(field_name);
            
            % Format field value based on type
            if ischar(field_value) || isstring(field_value)
                formatted_value = sprintf('"%s"', char(field_value));
            elseif isnumeric(field_value) && isscalar(field_value)
                if isnan(field_value)
                    formatted_value = 'null';
                else
                    formatted_value = sprintf('%.10g', field_value);
                end
            elseif islogical(field_value)
                formatted_value = sprintf('%s', lower(char(string(field_value))));
            else
                formatted_value = '"unknown"';
            end
            
            json_parts{end+1} = sprintf('"%s": %s', field_name, formatted_value);
        end
        json_str = sprintf('{%s}', strjoin(json_parts, ', '));
    else
        json_str = '{}';
    end
end

function result = octave_startsWith(str, prefix)
    % String prefix checking with Octave fallback
    if exist('startsWith', 'builtin') || exist('startsWith', 'file')
        result = startsWith(str, prefix);
    else
        str = char(str);
        prefix = char(prefix);
        result = length(str) >= length(prefix) && strcmp(str(1:length(prefix)), prefix);
    end
end

function result = octave_endsWith(str, suffix)
    % String suffix checking with Octave fallback
    if exist('endsWith', 'builtin') || exist('endsWith', 'file')
        result = endsWith(str, suffix);
    else
        str = char(str);
        suffix = char(suffix);
        result = length(str) >= length(suffix) && strcmp(str(end-length(suffix)+1:end), suffix);
    end
end

function result = octave_contains(str, pattern)
    % String contains checking with Octave fallback
    if exist('contains', 'builtin') || exist('contains', 'file')
        result = contains(str, pattern);
    else
        result = ~isempty(strfind(str, pattern));
    end
end
```

**Replace all function calls throughout the file:**
- `jsonencode(...)` → `octave_jsonencode(...)`
- `startsWith(...)` → `octave_startsWith(...)`
- `endsWith(...)` → `octave_endsWith(...)`
- `contains(...)` → `octave_contains(...)`

**Critical locations to update:**
- Line ~357: `if octave_contains(line, 'problem_libraries:')`
- Line ~368: `~octave_contains(line, '#')`
- Line ~387: `if octave_contains(attr_line, ':')`

---

## Step-by-Step Troubleshooting Process

### Phase 1: Initial Setup and Compilation

1. **Verify Submodules and Architecture Support**
   ```bash
   # Check git submodules are properly initialized
   git submodule status
   
   # Test basic Octave functionality
   ./scripts/docker-run.sh --octave -- octave --no-gui --eval "disp('Octave working')"
   ```

2. **Add ARM64 Architecture Support**
   - Update both `install_sedumi.m` and `install_sdpt3.m`
   - Test compilation with setup script

3. **Verify MEX Compilation**
   ```bash
   ./scripts/docker-run.sh --octave -- octave --no-gui --eval "setup_matlab_solvers"
   ```

### Phase 2: Runtime Compatibility

4. **Fix GUI-Related Issues**
   - Add `--no-gui` flag to Octave commands
   - Test basic solver execution

5. **Implement Function Compatibility**
   - Add compatibility shims to `matlab_solver_runner.m`
   - Replace all incompatible function calls
   - Test incrementally after each function replacement

### Phase 3: End-to-End Testing

6. **Test Individual Solvers**
   ```bash
   # Test SDPT3
   ./scripts/docker-run.sh --octave -- python main.py --benchmark --problems nb --solvers matlab_sdpt3 --dry-run
   
   # Test SeDuMi
   ./scripts/docker-run.sh --octave -- python main.py --benchmark --problems nb --solvers matlab_sedumi --dry-run
   ```

7. **Verify Results**
   - Check for OPTIMAL status
   - Verify reasonable solve times
   - Confirm objective values match between solvers

---

## Common Error Patterns and Solutions

### Architecture Errors
```
Error: Unexpected computer type: aarch64-unknown-linux-gnu
```
**Fix:** Add architecture case to installation scripts

### GUI Errors
```
octave: X11 DISPLAY environment variable not set
MATLAB subprocess failed (code 1)
```
**Fix:** Add `--no-gui` flag to Octave commands

### Function Compatibility Errors
```
Error: 'contains' undefined near line 357, column 357
```
**Fix:** Replace with `octave_contains` and ensure compatibility function is defined

### JSON Encoding Errors
```
Error: 'jsonencode' undefined
```
**Fix:** Use `octave_jsonencode` with custom fallback implementation

---

## Performance Benchmarks

**Test Problem:** `nb` (SOCP, 2383 variables, 123 constraints)

| Solver | Status | Solve Time | Objective Value |
|--------|--------|------------|-----------------|
| SeDuMi | OPTIMAL | 1.083s | -5.070309e-02 |
| SDPT3 | OPTIMAL | 4.092s | -5.070309e-02 |

**Success Rate:** 100% for supported problem types

---

## Docker Environment Considerations

### Required Packages
The Docker environment includes these essential packages for MEX compilation:
```dockerfile
# Octave for MATLAB compatibility with MEX compilation support
octave \
liboctave-dev \
# Build tools for MEX compilation
build-essential \
gcc \
g++ \
gfortran \
libopenblas-dev \
liblapack-dev \
libblas-dev \
```

### Environment Variables
```bash
MATLAB_SOLVER_ENGINE=octave  # Enable Octave mode
DISPLAY=''                   # Disable X11
_JAVA_OPTIONS='-Djava.awt.headless=true'  # Headless Java
```

---

## Future Maintenance

### Adding New MATLAB Functions
When Octave lacks a MATLAB function:

1. **Create compatibility function in `matlab_solver_runner.m`:**
   ```matlab
   function result = octave_new_function(args)
       if exist('new_function', 'builtin') || exist('new_function', 'file')
           result = new_function(args);
       else
           % Implement fallback logic
           result = fallback_implementation(args);
       end
   end
   ```

2. **Replace all calls throughout the codebase**
3. **Test with representative problems**

### Adding New Architectures
1. Add architecture case to installation scripts
2. Test MEX compilation
3. Verify solver execution

### Debugging Process
1. **Test individual components:**
   - Octave installation and basic functionality
   - MEX compilation for each solver
   - Direct solver execution via command line
   - Python interface integration

2. **Use verbose logging:**
   ```bash
   # Enable debug logging
   PYTHONPATH=. python -c "
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Run benchmark with detailed output
   "
   ```

---

## Conclusion

Successfully running SeDuMi and SDPT3 with Octave requires addressing three main categories of issues:

1. **Architecture Compatibility** - Adding ARM64 support to installation scripts
2. **Environment Configuration** - Using appropriate command-line flags for headless operation  
3. **Function Compatibility** - Implementing fallback functions for MATLAB-specific features

This setup provides a robust, license-free alternative to MATLAB for optimization solver benchmarking while maintaining full compatibility with the existing benchmark system architecture.

**Key Success Factors:**
- Systematic approach to identifying and fixing compatibility issues
- Comprehensive testing at each step
- Proper documentation of changes for future maintenance

The resulting system achieves 100% success rate for supported problem types with performance comparable to native MATLAB execution.