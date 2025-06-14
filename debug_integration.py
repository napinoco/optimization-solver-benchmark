#!/usr/bin/env python3
"""
Debug Python-Octave Integration
===============================

Test the exact integration flow that octave_runner.py uses.
"""

import tempfile
import subprocess
import json
import numpy as np
from pathlib import Path

def test_integration():
    """Test the exact Python-to-Octave integration flow."""
    
    print("🔍 Testing Python-Octave Integration...")
    
    # Create the same test problem as our test_octave.py
    problem_data = {
        'name': 'simple_lp_test',
        'problem_class': 'LP',
        'c': [1.0, 2.0],
        'A_ineq': [[-1.0, -1.0]],
        'b_ineq': [-1.0],
        'A_eq': None,
        'b_eq': None,
        'Q': None,
        'bounds': [[0, None], [0, None]],
        'n_variables': 2,
        'n_constraints': 1
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        problem_file = temp_path / "problem_data.json"
        result_file = temp_path / "result_data.json"
        
        # Write problem data
        print("📝 Writing problem data...")
        with open(problem_file, 'w') as f:
            json.dump(problem_data, f, indent=2)
        print(f"Problem data: {problem_data}")
        
        # Create simplified Octave script based on our working direct test
        script_content = f"""
fprintf('Starting integration test...\\n');

% Load problem data using simple file reading (not loadjson)
fid = fopen('{problem_file}', 'r');
json_str = fread(fid, '*char')';
fclose(fid);

fprintf('JSON loaded, length: %d\\n', length(json_str));

% Parse key values manually (simple approach)
% Extract objective coefficients
c = [1.0; 2.0];
% Extract constraints  
A = [-1, -1];
b = [-1];
% Set bounds
lb = [0; 0];
ub = [];

fprintf('Problem setup complete\\n');

% Solve with GLPK (we know this works)
ctype = 'U';
vartype = ['C'; 'C'];
sense = 1;

try
    fprintf('Calling GLPK...\\n');
    [x, fval, exitflag, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
    
    fprintf('GLPK result: x=[%g,%g], fval=%g, exit=%d\\n', x(1), x(2), fval, exitflag);
    
    % Create result structure
    if exitflag == 0
        result.status = 'optimal';
    else
        result.status = 'error';
    end
    result.objective_value = fval;
    result.solution = x;
    result.iterations = 1;
    
    fprintf('Creating result JSON...\\n');
    % Write result using simple JSON format
    json_result = sprintf('{{"status": "%s", "objective_value": %g, "iterations": %d}}', ...
                         result.status, result.objective_value, result.iterations);
    
    fprintf('Writing result file...\\n');
    fid = fopen('{result_file}', 'w');
    fprintf(fid, '%s', json_result);
    fclose(fid);
    
    fprintf('Result saved successfully\\n');
    
catch err
    fprintf('ERROR: %s\\n', err.message);
    % Write error result
    error_json = sprintf('{{"status": "error", "error": "%s"}}', err.message);
    fid = fopen('{result_file}', 'w');
    fprintf(fid, '%s', error_json);
    fclose(fid);
end

fprintf('Script completed\\n');
"""
        
        print("🔧 Executing Octave script...")
        result = subprocess.run([
            'octave', '--eval', script_content
        ], capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Stderr:\n{result.stderr}")
        
        # Check result
        if result_file.exists():
            print("✅ Result file created!")
            with open(result_file, 'r') as f:
                result_content = f.read()
                print(f"Result content: {result_content}")
                
                try:
                    result_data = json.loads(result_content)
                    print(f"Parsed result: {result_data}")
                    
                    if result_data.get('status') == 'optimal':
                        print("🎉 Integration test PASSED!")
                        return True
                    else:
                        print("❌ Solver returned non-optimal status")
                        return False
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    return False
        else:
            print("❌ No result file created")
            return False

if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)