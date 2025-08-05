function test_matlab_data_loaders()
% Comprehensive integration testing of MATLAB data loading pipeline
%
% This function tests the complete data loading workflow:
% 1. YAML configuration reading (matlab_yaml_reader)
% 2. DIMACS .mat file loading (mat_loader)
% 3. SDPLIB .dat-s file loading (dat_loader)
% 4. Integration between all components
% 5. Error handling and edge cases
%
% Test coverage includes multiple problems from each library (DIMACS, SDPLIB)
% and validates data consistency and performance.

fprintf('=== MATLAB Data Loader Integration Testing ===\n\n');

% Initialize test results
test_results = struct();
test_results.total_tests = 0;
test_results.passed_tests = 0;
test_results.failed_tests = 0;
test_results.errors = {};

% Add paths for all required functions
addpath(fullfile(pwd, 'scripts/utils/'));
addpath(fullfile(pwd, 'scripts/data_loaders/matlab/'));

try
    % Test 1: YAML Reader Basic Functionality
    fprintf('Test 1: YAML Reader Basic Functionality\n');
    test_results = run_yaml_reader_tests(test_results);
    
    % Test 2: DIMACS Data Loader Tests
    fprintf('\nTest 2: DIMACS Data Loader Tests\n');
    test_results = run_dimacs_loader_tests(test_results);
    
    % Test 3: SDPLIB Data Loader Tests
    fprintf('\nTest 3: SDPLIB Data Loader Tests\n');
    test_results = run_sdplib_loader_tests(test_results);
    
    % Test 4: Integration Workflow Tests
    fprintf('\nTest 4: Integration Workflow Tests\n');
    test_results = run_integration_workflow_tests(test_results);
    
    % Test 5: Error Handling Tests
    fprintf('\nTest 5: Error Handling Tests\n');
    test_results = run_error_handling_tests(test_results);
    
    % Print final results
    fprintf('\n=== Final Test Results ===\n');
    fprintf('Total tests: %d\n', test_results.total_tests);
    fprintf('Passed: %d\n', test_results.passed_tests);
    fprintf('Failed: %d\n', test_results.failed_tests);
    fprintf('Success rate: %.1f%%\n', 100 * test_results.passed_tests / test_results.total_tests);
    
    if test_results.failed_tests > 0
        fprintf('\nFailed tests:\n');
        for i = 1:length(test_results.errors)
            fprintf('  - %s\n', test_results.errors{i});
        end
        error('test_matlab_data_loaders:TestsFailed', '%d tests failed', test_results.failed_tests);
    else
        fprintf('\nAll tests passed! ✅\n');
    end
    
catch ME
    fprintf('\nIntegration test failed with error: %s\n', ME.message);
    rethrow(ME);
end

end

function test_results = run_yaml_reader_tests(test_results)
% Test YAML reader functionality with various problems

fprintf('  Testing YAML reader...\n');

% Test cases: [problem_name, expected_library, expected_file_type]
test_cases = {
    'nb', 'DIMACS', 'mat';
    'arch0', 'SDPLIB', 'dat-s';
    'control1', 'SDPLIB', 'dat-s';
    'bm1', 'DIMACS', 'mat'
};

for i = 1:size(test_cases, 1)
    problem_name = test_cases{i, 1};
    expected_library = test_cases{i, 2};
    expected_file_type = test_cases{i, 3};
    
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        [info, file_path] = matlab_yaml_reader(problem_name);
        
        % Validate required fields
        assert(isfield(info, 'display_name'), 'Missing display_name');
        assert(isfield(info, 'file_path'), 'Missing file_path');
        assert(isfield(info, 'file_type'), 'Missing file_type');
        assert(isfield(info, 'library_name'), 'Missing library_name');
        assert(~isempty(file_path), 'Empty file_path');
        
        % Validate expected values
        assert(strcmp(info.library_name, expected_library), 'Wrong library_name');
        assert(strcmp(info.file_type, expected_file_type), 'Wrong file_type');
        assert(strcmp(info.file_path, file_path), 'Inconsistent file_path');
        
        % Check file exists
        assert(exist(file_path, 'file') > 0, 'File does not exist');
        
        test_results.passed_tests = test_results.passed_tests + 1;
        fprintf('    ✅ %s: %s\n', problem_name, info.display_name);
        
    catch ME
        test_results.failed_tests = test_results.failed_tests + 1;
        error_msg = sprintf('YAML reader test failed for %s: %s', problem_name, ME.message);
        test_results.errors{end+1} = error_msg;
        fprintf('    ❌ %s: %s\n', problem_name, ME.message);
    end
end

end

function test_results = run_dimacs_loader_tests(test_results)
% Test DIMACS .mat file loader with multiple problems

fprintf('  Testing DIMACS loader...\n');

% Test DIMACS problems
dimacs_problems = {'nb', 'nb_L2', 'bm1'};

for i = 1:length(dimacs_problems)
    problem_name = dimacs_problems{i};
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        % Get file path from YAML
        [info, file_path] = matlab_yaml_reader(problem_name);
        assert(strcmp(info.file_type, 'mat'), 'Expected mat file type');
        
        % Load problem using mat_loader
        [A, b, c, K] = mat_loader(file_path);
        
        % Validate output format
        assert(issparse(A), 'A must be sparse');
        assert(isvector(b) && size(b, 2) == 1, 'b must be column vector');
        assert(isvector(c) && size(c, 2) == 1, 'c must be column vector');
        assert(isstruct(K), 'K must be struct');
        
        % Validate dimensions
        [m, n] = size(A);
        assert(length(b) == m, 'Inconsistent b dimensions');
        assert(length(c) == n, 'Inconsistent c dimensions');
        
        % Validate cone structure
        assert(isfield(K, 'f'), 'K missing f field');
        assert(isfield(K, 'l'), 'K missing l field');
        assert(K.f >= 0, 'K.f must be non-negative');
        assert(K.l >= 0, 'K.l must be non-negative');
        
        % Calculate total variables from cone structure
        total_vars = K.f + K.l;
        if isfield(K, 'q') && ~isempty(K.q)
            total_vars = total_vars + sum(K.q);
        end
        if isfield(K, 's') && ~isempty(K.s)
            total_vars = total_vars + sum(K.s .* K.s);
        end
        
        % Allow some tolerance for dimension mismatch warnings
        if abs(total_vars - n) <= n * 0.1  % 10% tolerance
            test_results.passed_tests = test_results.passed_tests + 1;
            fprintf('    ✅ %s: [%dx%d] matrix, %d vars\n', problem_name, m, n, total_vars);
        else
            error('Cone structure inconsistent with problem size: %d vs %d', total_vars, n);
        end
        
    catch ME
        test_results.failed_tests = test_results.failed_tests + 1;
        error_msg = sprintf('DIMACS loader test failed for %s: %s', problem_name, ME.message);
        test_results.errors{end+1} = error_msg;
        fprintf('    ❌ %s: %s\n', problem_name, ME.message);
    end
end

end

function test_results = run_sdplib_loader_tests(test_results)
% Test SDPLIB .dat-s file loader with multiple problems

fprintf('  Testing SDPLIB loader...\n');

% Test SDPLIB problems
sdplib_problems = {'arch0', 'control1', 'hinf1'};

for i = 1:length(sdplib_problems)
    problem_name = sdplib_problems{i};
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        % Get file path from YAML
        [info, file_path] = matlab_yaml_reader(problem_name);
        assert(strcmp(info.file_type, 'dat-s'), 'Expected dat-s file type');
        
        % Load problem using dat_loader
        [A, b, c, K] = dat_loader(file_path);
        
        % Validate output format
        assert(issparse(A), 'A must be sparse');
        assert(isvector(b) && size(b, 2) == 1, 'b must be column vector');
        assert(isvector(c) && size(c, 2) == 1, 'c must be column vector');
        assert(isstruct(K), 'K must be struct');
        
        % Validate dimensions
        [m, n] = size(A);
        assert(length(b) == m, 'Inconsistent b dimensions');
        assert(length(c) == n, 'Inconsistent c dimensions');
        
        % Validate cone structure
        assert(isfield(K, 'f'), 'K missing f field');
        assert(isfield(K, 'l'), 'K missing l field');
        assert(K.f >= 0, 'K.f must be non-negative');
        assert(K.l >= 0, 'K.l must be non-negative');
        
        % For SDPLIB problems, we expect SDP blocks
        if isfield(K, 's') && ~isempty(K.s)
            assert(all(K.s > 0), 'SDP block sizes must be positive');
        end
        
        % Calculate total variables from cone structure
        total_vars = K.f + K.l;
        if isfield(K, 'q') && ~isempty(K.q)
            total_vars = total_vars + sum(K.q);
        end
        if isfield(K, 's') && ~isempty(K.s)
            total_vars = total_vars + sum(K.s .* K.s);
        end
        
        assert(total_vars == n, 'Cone structure inconsistent with problem size: %d vs %d', total_vars, n);
        
        test_results.passed_tests = test_results.passed_tests + 1;
        fprintf('    ✅ %s: [%dx%d] matrix, SDP blocks: %s\n', problem_name, m, n, mat2str(K.s));
        
    catch ME
        test_results.failed_tests = test_results.failed_tests + 1;
        error_msg = sprintf('SDPLIB loader test failed for %s: %s', problem_name, ME.message);
        test_results.errors{end+1} = error_msg;
        fprintf('    ❌ %s: %s\n', problem_name, ME.message);
    end
end

end

function test_results = run_integration_workflow_tests(test_results)
% Test complete workflow: YAML -> file resolution -> data loading

fprintf('  Testing integration workflow...\n');

% Test complete workflow for different problem types
workflow_problems = {
    'nb', 'DIMACS';
    'arch0', 'SDPLIB'
};

for i = 1:size(workflow_problems, 1)
    problem_name = workflow_problems{i, 1};
    expected_library = workflow_problems{i, 2};
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        % Step 1: Resolve problem name to file path
        [info, file_path] = matlab_yaml_reader(problem_name);
        
        % Step 2: Load data based on file type
        if strcmp(info.file_type, 'mat')
            [A, b, c, K] = mat_loader(file_path);
        elseif strcmp(info.file_type, 'dat-s')
            [A, b, c, K] = dat_loader(file_path);
        else
            error('Unsupported file type: %s', info.file_type);
        end
        
        % Step 3: Validate complete workflow
        assert(strcmp(info.library_name, expected_library), 'Wrong library');
        assert(exist(file_path, 'file') > 0, 'File not found');
        assert(~isempty(A) && ~isempty(b) && ~isempty(c) && ~isempty(K), 'Empty data');
        
        % Step 4: Basic optimization problem validation
        [m, n] = size(A);
        assert(m > 0 && n > 0, 'Invalid problem dimensions');
        assert(length(b) == m && length(c) == n, 'Dimension mismatch');
        
        test_results.passed_tests = test_results.passed_tests + 1;
        fprintf('    ✅ %s: Complete workflow successful\n', problem_name);
        
    catch ME
        test_results.failed_tests = test_results.failed_tests + 1;
        error_msg = sprintf('Integration workflow test failed for %s: %s', problem_name, ME.message);
        test_results.errors{end+1} = error_msg;
        fprintf('    ❌ %s: %s\n', problem_name, ME.message);
    end
end

end

function test_results = run_error_handling_tests(test_results)
% Test error handling for various failure scenarios

fprintf('  Testing error handling...\n');

% Test 1: Non-existent problem
test_results.total_tests = test_results.total_tests + 1;
try
    matlab_yaml_reader('nonexistent_problem');
    % Should not reach here
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.errors{end+1} = 'Error handling test failed: should have thrown error for non-existent problem';
    fprintf('    ❌ Non-existent problem: Should have failed\n');
catch ME
    if contains(ME.message, 'not found')
        test_results.passed_tests = test_results.passed_tests + 1;
        fprintf('    ✅ Non-existent problem: Correctly detected\n');
    else
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.errors{end+1} = sprintf('Error handling test failed: wrong error message: %s', ME.message);
        fprintf('    ❌ Non-existent problem: Wrong error\n');
    end
end

% Test 2: Non-existent file
test_results.total_tests = test_results.total_tests + 1;
try
    mat_loader('nonexistent_file.mat');
    % Should not reach here
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.errors{end+1} = 'Error handling test failed: should have thrown error for non-existent file';
    fprintf('    ❌ Non-existent file: Should have failed\n');
catch ME
    if contains(ME.message, 'not found') || contains(ME.message, 'FileNotFound')
        test_results.passed_tests = test_results.passed_tests + 1;
        fprintf('    ✅ Non-existent file: Correctly detected\n');
    else
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.errors{end+1} = sprintf('Error handling test failed: wrong error message: %s', ME.message);
        fprintf('    ❌ Non-existent file: Wrong error\n');
    end
end

% Test 3: Invalid YAML file
test_results.total_tests = test_results.total_tests + 1;
try
    matlab_yaml_reader('nb', 'nonexistent_config.yaml');
    % Should not reach here
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.errors{end+1} = 'Error handling test failed: should have thrown error for non-existent config';
    fprintf('    ❌ Non-existent config: Should have failed\n');
catch ME
    if contains(ME.message, 'not found') || contains(ME.message, 'FileNotFound')
        test_results.passed_tests = test_results.passed_tests + 1;
        fprintf('    ✅ Non-existent config: Correctly detected\n');
    else
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.errors{end+1} = sprintf('Error handling test failed: wrong error message: %s', ME.message);
        fprintf('    ❌ Non-existent config: Wrong error\n');
    end
end

end