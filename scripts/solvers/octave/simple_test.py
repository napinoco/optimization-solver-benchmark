#!/usr/bin/env python3
"""
Simple Octave Integration Test
=============================

Minimal test to verify Octave can execute and return basic results.
"""

import tempfile
import subprocess
import json
from pathlib import Path

def test_basic_octave():
    """Test basic Octave execution and JSON I/O."""
    
    print("🔍 Testing Basic Octave Integration...")
    
    try:
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.json"
            output_file = temp_path / "output.json"
            
            # Write test input
            test_data = {"x": [1, 2], "y": 3}
            with open(input_file, 'w') as f:
                json.dump(test_data, f)
            
            # Create simple Octave script
            script_content = f"""
% Simple test script
fprintf('Octave is running...\\n');

% Test basic math
x = [1, 2];
y = 3;
result = sum(x) + y;

% Create result structure
output.status = 'success';
output.result = result;
output.message = 'Basic Octave test completed';

% Convert to JSON string manually (simple approach)
json_str = sprintf('{{"status": "%s", "result": %g, "message": "%s"}}', ...
                   output.status, output.result, output.message);

% Write to file
fid = fopen('{str(output_file)}', 'w');
fprintf(fid, '%s', json_str);
fclose(fid);

fprintf('Result saved to file\\n');
"""
            
            # Execute Octave
            print("🔧 Executing Octave script...")
            result = subprocess.run([
                'octave', '--eval', script_content
            ], capture_output=True, text=True, timeout=30)
            
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            
            # Check if output file was created
            if output_file.exists():
                print("✅ Output file created successfully!")
                with open(output_file, 'r') as f:
                    output_data = json.load(f)
                print(f"Output: {output_data}")
                
                if output_data.get('status') == 'success' and output_data.get('result') == 6:
                    print("✅ Basic Octave integration test PASSED!")
                    return True
                else:
                    print("❌ Unexpected result values")
                    return False
            else:
                print("❌ Output file was not created")
                return False
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_octave()
    exit(0 if success else 1)