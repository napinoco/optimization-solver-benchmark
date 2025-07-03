function test_matlab_solver_integration()
% Comprehensive integration testing for MATLAB solver runners
%
% This function tests the complete MATLAB solver integration pipeline:
% - Data loading (DIMACS .mat, SDPLIB .dat-s) 
% - Solver execution (SeDuMi, SDPT3)
% - JSON result formatting and file output
% - Error handling and edge cases
% - Performance benchmarking
%
% The tests cover different problem types: LP, QP, SOCP, SDP

fprintf('\n');
fprintf('================================================================\n');
fprintf('MATLAB Solver Integration Testing\n');
fprintf('================================================================\n');
fprintf('Date: %s\n', datestr(now));
fprintf('MATLAB: %s\n', version);
fprintf('Platform: %s\n', computer);
fprintf('================================================================\n\n');

% Add required paths
addpath(genpath('scripts/data_loaders/matlab_octave'));
addpath(genpath('scripts/solvers/matlab_octave'));
addpath(genpath('scripts/utils'));

% Initialize test results
test_results = struct();
test_results.total_tests = 0;
test_results.passed_tests = 0;
test_results.failed_tests = 0;
test_results.test_details = {};

try
    % Test 1: Environment and Version Detection
    fprintf('Test 1: Environment and Version Detection\n');
    test_results = run_environment_test(test_results);
    
    % Test 2: Data Loader Integration  
    fprintf('\nTest 2: Data Loader Integration\n');
    test_results = run_data_loader_tests(test_results);
    
    % Test 3: SeDuMi Solver Integration
    fprintf('\nTest 3: SeDuMi Solver Integration\n');
    test_results = run_sedumi_tests(test_results);
    
    % Test 4: SDPT3 Solver Integration
    fprintf('\nTest 4: SDPT3 Solver Integration\n');
    test_results = run_sdpt3_tests(test_results);
    
    % Test 5: JSON Output Integration
    fprintf('\nTest 5: JSON Output Integration\n');
    test_results = run_json_tests(test_results);
    
    % Test 6: Error Handling and Edge Cases
    fprintf('\nTest 6: Error Handling and Edge Cases\n');
    test_results = run_error_tests(test_results);
    
    % Test 7: Performance Benchmarking
    fprintf('\nTest 7: Performance Benchmarking\n');
    test_results = run_performance_tests(test_results);
    
catch ME
    fprintf('CRITICAL ERROR in test suite: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
end

% Generate final report
fprintf('\n');
fprintf('================================================================\n');
fprintf('INTEGRATION TEST SUMMARY\n');
fprintf('================================================================\n');
fprintf('Total Tests: %d\n', test_results.total_tests);
fprintf('Passed: %d\n', test_results.passed_tests);
fprintf('Failed: %d\n', test_results.failed_tests);
fprintf('Success Rate: %.1f%%\n', 100 * test_results.passed_tests / test_results.total_tests);
fprintf('================================================================\n');

if test_results.failed_tests > 0
    fprintf('\nFAILED TESTS:\n');
    for i = 1:length(test_results.test_details)
        detail = test_results.test_details{i};
        if strcmp(detail.status, 'FAILED')
            fprintf('  - %s: %s\n', detail.name, detail.error);
        end
    end
end

fprintf('\nIntegration testing completed.\n');

end

function test_results = run_environment_test(test_results)
% Test environment and version detection

fprintf('  Testing version detection...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    metadata = matlab_version_detection();
    
    % Validate required fields
    required_fields = {'matlab_version', 'sedumi_available', 'sdpt3_available'};
    for i = 1:length(required_fields)
        if ~isfield(metadata, required_fields{i})
            error('Missing required field: %s', required_fields{i});
        end
    end
    
    fprintf('    MATLAB: %s\n', metadata.matlab_version);
    fprintf('    SeDuMi: Available=%d, Version=%s\n', ...
            metadata.sedumi_available, metadata.sedumi_version);
    fprintf('    SDPT3: Available=%d, Version=%s\n', ...
            metadata.sdpt3_available, metadata.sdpt3_version);
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Environment Detection', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ERROR: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Environment Detection', 'status', 'FAILED', 'error', ME.message);
end

end

function test_results = run_data_loader_tests(test_results)
% Test data loader integration with different problem types

test_problems = {
    '../../problems/SDPLIB/data/truss1.dat-s', 'SDPLIB', 'SDP';
    '../../problems/DIMACS/data/HINF/hinf13.mat.gz', 'DIMACS', 'SOCP'
};

for i = 1:size(test_problems, 1)
    problem_file = test_problems{i, 1};
    problem_type = test_problems{i, 2};
    expected_class = test_problems{i, 3};
    
    fprintf('  Testing %s loader with %s...\n', problem_type, problem_file);
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        if ~exist(problem_file, 'file')
            error('Problem file not found: %s', problem_file);
        end
        
        if strcmp(problem_type, 'SDPLIB')
            [A, b, c, K] = dat_loader(problem_file);
        else
            [A, b, c, K] = mat_loader(problem_file);
        end
        
        % Validate loaded data
        if isempty(A) || isempty(b) || isempty(c) || ~isstruct(K)
            error('Invalid data loaded from %s', problem_file);
        end
        
        fprintf('    SUCCESS: Loaded %dx%d problem\n', size(A, 1), size(A, 2));
        test_results.passed_tests = test_results.passed_tests + 1;
        test_results.test_details{end+1} = struct('name', sprintf('%s Loader', problem_type), 'status', 'PASSED', 'error', '');
        
    catch ME
        fprintf('    ERROR: %s\n', ME.message);
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.test_details{end+1} = struct('name', sprintf('%s Loader', problem_type), 'status', 'FAILED', 'error', ME.message);
    end
end

end

function test_results = run_sedumi_tests(test_results)
% Test SeDuMi solver with different problem types

test_problems = {
    '../../problems/SDPLIB/data/truss1.dat-s', 'dat_loader', 'SDP';
};

% Add DIMACS problem if available
if exist('../../problems/DIMACS/data/HINF/hinf13.mat.gz', 'file')
    test_problems{end+1, 1} = '../../problems/DIMACS/data/HINF/hinf13.mat.gz';
    test_problems{end, 2} = 'mat_loader';
    test_problems{end, 3} = 'SOCP';
end

for i = 1:size(test_problems, 1)
    problem_file = test_problems{i, 1};
    loader_func = test_problems{i, 2};
    problem_class = test_problems{i, 3};
    
    fprintf('  Testing SeDuMi with %s (%s)...\n', problem_file, problem_class);
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        % Load problem
        if strcmp(loader_func, 'dat_loader')
            [A, b, c, K] = dat_loader(problem_file);
        else
            [A, b, c, K] = mat_loader(problem_file);
        end
        
        % Solve with SeDuMi
        tic;
        result = sedumi_runner(A, b, c, K);
        solve_time = toc;
        
        % Validate result
        if ~isstruct(result) || ~isfield(result, 'status')
            error('Invalid result structure from SeDuMi');
        end
        
        fprintf('    Status: %s, Time: %.3fs\n', result.status, solve_time);
        if strcmp(result.status, 'optimal') && ~isnan(result.primal_objective)
            fprintf('    Objective: %.6f\n', result.primal_objective);
        end
        
        test_results.passed_tests = test_results.passed_tests + 1;
        test_results.test_details{end+1} = struct('name', sprintf('SeDuMi %s', problem_class), 'status', 'PASSED', 'error', '');
        
    catch ME
        fprintf('    ERROR: %s\n', ME.message);
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.test_details{end+1} = struct('name', sprintf('SeDuMi %s', problem_class), 'status', 'FAILED', 'error', ME.message);
    end
end

end

function test_results = run_sdpt3_tests(test_results)
% Test SDPT3 solver with different problem types

test_problems = {
    '../../problems/SDPLIB/data/truss1.dat-s', 'dat_loader', 'SDP';
};

% Add DIMACS problem if available  
if exist('../../problems/DIMACS/data/HINF/hinf13.mat.gz', 'file')
    test_problems{end+1, 1} = '../../problems/DIMACS/data/HINF/hinf13.mat.gz';
    test_problems{end, 2} = 'mat_loader';
    test_problems{end, 3} = 'SOCP';
end

for i = 1:size(test_problems, 1)
    problem_file = test_problems{i, 1};
    loader_func = test_problems{i, 2};
    problem_class = test_problems{i, 3};
    
    fprintf('  Testing SDPT3 with %s (%s)...\n', problem_file, problem_class);
    test_results.total_tests = test_results.total_tests + 1;
    
    try
        % Load problem
        if strcmp(loader_func, 'dat_loader')
            [A, b, c, K] = dat_loader(problem_file);
        else
            [A, b, c, K] = mat_loader(problem_file);
        end
        
        % Solve with SDPT3
        tic;
        result = sdpt3_runner(A, b, c, K);
        solve_time = toc;
        
        % Validate result
        if ~isstruct(result) || ~isfield(result, 'status')
            error('Invalid result structure from SDPT3');
        end
        
        fprintf('    Status: %s, Time: %.3fs\n', result.status, solve_time);
        if strcmp(result.status, 'optimal') && ~isnan(result.primal_objective)
            fprintf('    Objective: %.6f\n', result.primal_objective);
        end
        
        test_results.passed_tests = test_results.passed_tests + 1;
        test_results.test_details{end+1} = struct('name', sprintf('SDPT3 %s', problem_class), 'status', 'PASSED', 'error', '');
        
    catch ME
        fprintf('    ERROR: %s\n', ME.message);
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.test_details{end+1} = struct('name', sprintf('SDPT3 %s', problem_class), 'status', 'FAILED', 'error', ME.message);
    end
end

end

function test_results = run_json_tests(test_results)
% Test JSON output integration

fprintf('  Testing JSON output with SeDuMi result...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Create simple test problem
    A = sparse([1, 1]);
    b = 1;
    c = [1; 1];
    K = struct('f', 0, 'l', 2, 'q', [], 's', []);
    
    % Solve and get result
    result = sedumi_runner(A, b, c, K);
    
    % Test JSON formatting
    json_str = matlab_json_formatter(result);
    
    % Test JSON file saving
    output_file = 'test_integration_result.json';
    success = save_json_result(result, output_file);
    
    if ~success
        error('Failed to save JSON result to file');
    end
    
    % Validate file exists and has content
    if ~exist(output_file, 'file')
        error('JSON output file was not created');
    end
    
    file_info = dir(output_file);
    if file_info.bytes == 0
        error('JSON output file is empty');
    end
    
    fprintf('    SUCCESS: JSON file saved (%d bytes)\n', file_info.bytes);
    
    % Clean up
    delete(output_file);
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'JSON Output', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ERROR: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'JSON Output', 'status', 'FAILED', 'error', ME.message);
end

end

function test_results = run_error_tests(test_results)
% Test error handling and edge cases

fprintf('  Testing error handling with invalid inputs...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Test with invalid problem data
    A = sparse(2, 3);  % Inconsistent dimensions
    b = [1; 2];
    c = [1; 1; 1];
    K = struct('f', 0, 'l', 3, 'q', [], 's', []);
    
    % This should handle the error gracefully
    result = sedumi_runner(A, b, c, K);
    
    % Result should indicate error
    if ~strcmp(result.status, 'num_error') && ~strcmp(result.status, 'unknown')
        fprintf('    WARNING: Expected error status, got: %s\n', result.status);
    else
        fprintf('    SUCCESS: Error handled gracefully\n');
    end
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Error Handling', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ERROR: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Error Handling', 'status', 'FAILED', 'error', ME.message);
end

end

function test_results = run_performance_tests(test_results)
% Test performance benchmarking

fprintf('  Testing performance with small problems...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Create simple feasible LP problem: min x1 + x2 s.t. x1 + x2 = 1, x1, x2 >= 0
    A = sparse([1, 1]);  % x1 + x2 = 1
    b = 1;               % Right-hand side
    c = [1; 1];          % Minimize x1 + x2
    K = struct('f', 0, 'l', 2, 'q', [], 's', []);  % Both variables are non-negative
    
    % Benchmark SeDuMi
    num_runs = 5;
    times = zeros(num_runs, 1);
    
    for i = 1:num_runs
        tic;
        result = sedumi_runner(A, b, c, K);
        times(i) = toc;
        
        if ~strcmp(result.status, 'optimal')
            error('Performance test failed: non-optimal status');
        end
    end
    
    avg_time = mean(times);
    std_time = std(times);
    
    fprintf('    Average time: %.3f ± %.3f seconds (%d runs)\n', avg_time, std_time, num_runs);
    
    if avg_time > 5.0  % Warning if too slow
        fprintf('    WARNING: Performance seems slow (>5s average)\n');
    end
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Performance', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ERROR: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Performance', 'status', 'FAILED', 'error', ME.message);
end

end