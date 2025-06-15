#!/usr/bin/env python3

import tempfile
import subprocess
import json
from pathlib import Path

# Test the exact GLPK call that our solver is making
script_content = """
fprintf('Starting GLPK test...\\n');

% Test data like our simple LP problem
c = [1.0; 2.0];
A = [-1, -1];  
b = [-1];
lb = [0; 0];
ub = [];
ctype = 'U';
vartype = ['C'; 'C'];
sense = 1;

fprintf('Calling GLPK...\\n');
try
    [x, fval, exitflag, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
    fprintf('GLPK result: x=[%g,%g], fval=%g, exit=%d\\n', x(1), x(2), fval, exitflag);
    
    % Create result structure
    result.status = 'optimal';
    result.objective_value = fval;
    result.solution = x;
    
    % Save result
    fprintf('Saving result...\\n');
    json_str = sprintf('{"status": "%s", "objective_value": %g}', result.status, result.objective_value);
    fid = fopen('/tmp/debug_result.json', 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    fprintf('Result saved successfully\\n');
    
catch err
    fprintf('GLPK error: %s\\n', err.message);
end

fprintf('Script completed\\n');
"""

print("🔧 Testing GLPK script...")
result = subprocess.run([
    'octave', '--eval', script_content
], capture_output=True, text=True, timeout=30)

print(f"Return code: {result.returncode}")
print(f"Stdout:\n{result.stdout}")
if result.stderr:
    print(f"Stderr:\n{result.stderr}")

# Check if result file was created
result_file = Path('/tmp/debug_result.json')
if result_file.exists():
    print("✅ Result file created!")
    with open(result_file, 'r') as f:
        print(f"Content: {f.read()}")
else:
    print("❌ No result file created")