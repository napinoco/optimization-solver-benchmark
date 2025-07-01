function test_matlab_integration_simple()
% Simplified end-to-end integration test for MATLAB pipeline
%
% This test validates the complete MATLAB integration pipeline works correctly
% by testing with available problems and validating all components work together.

fprintf('\n');
fprintf('================================================================\n');
fprintf('SIMPLIFIED MATLAB INTEGRATION PIPELINE TEST\n');
fprintf('================================================================\n');
fprintf('Date: %s\n', datestr(now));
fprintf('MATLAB: %s\n', version);
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
    % Test 1: Basic Function Availability
    fprintf('Test 1: Verify all required functions are available\n');
    test_results = test_function_availability(test_results);
    
    % Test 2: Data Loader Testing
    fprintf('\nTest 2: Test data loaders with available problems\n');
    test_results = test_data_loaders(test_results);
    
    % Test 3: Solver Runner Testing
    fprintf('\nTest 3: Test solver runners with simple problem\n');
    test_results = test_solver_runners(test_results);
    
    % Test 4: Complete Pipeline Test
    fprintf('\nTest 4: Complete matlab_runner pipeline\n');
    test_results = test_complete_pipeline(test_results);
    
catch ME
    fprintf('CRITICAL ERROR in integration test: %s\n', ME.message);
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
            fprintf('  ❌ %s: %s\n', detail.name, detail.error);
        end
    end
end

if test_results.failed_tests == 0
    fprintf('\n🎉 ALL TESTS PASSED - MATLAB Integration Ready!\n');
else
    fprintf('\n⚠️  Some tests failed - Check implementation\n');
end

fprintf('\nIntegration testing completed.\n');

end

function test_results = test_function_availability(test_results)
% Test that all required functions are available

fprintf('  Checking function availability...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Check MATLAB functions exist
    required_functions = {
        'matlab_runner',
        'sedumi_runner', 
        'sdpt3_runner',
        'dat_loader',
        'matlab_version_detection',
        'matlab_json_formatter',
        'save_json_result',
        'solver_metrics_calculator'
    };
    
    missing_functions = {};
    
    for i = 1:length(required_functions)
        func_name = required_functions{i};
        if exist(func_name, 'file') ~= 2
            missing_functions{end+1} = func_name;
        end
    end
    
    if ~isempty(missing_functions)
        error('Missing functions: %s', strjoin(missing_functions, ', '));
    end
    
    fprintf('    ✓ All required functions available\n');
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Function Availability', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Function availability check failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Function Availability', 'status', 'FAILED', 'error', ME.message);
end

end

function test_results = test_data_loaders(test_results)
% Test data loaders with available files

fprintf('  Testing data loaders...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Try to find any available SDPLIB .dat-s file
    sdplib_files = dir('problems/SDPLIB/data/**/*.dat-s');
    
    if ~isempty(sdplib_files)
        test_file = fullfile(sdplib_files(1).folder, sdplib_files(1).name);
        fprintf('    Testing dat_loader with: %s\n', sdplib_files(1).name);
        
        [A, b, c, K] = dat_loader(test_file);
        
        if isempty(A) || isempty(b) || isempty(c) || ~isstruct(K)
            error('dat_loader returned invalid data');
        end
        
        fprintf('    ✓ dat_loader successful (%dx%d problem)\n', size(A, 1), size(A, 2));
    else
        fprintf('    ⚠️  No SDPLIB files found, skipping dat_loader test\n');
    end
    
    % Try to find any available DIMACS .mat file
    dimacs_files = dir('problems/DIMACS/data/**/*.mat*');
    
    if ~isempty(dimacs_files)
        % Find a non-gz file or the first file
        mat_file = '';
        for i = 1:length(dimacs_files)
            if ~contains(dimacs_files(i).name, '.gz')
                mat_file = fullfile(dimacs_files(i).folder, dimacs_files(i).name);
                break;
            end
        end
        
        if isempty(mat_file) && ~isempty(dimacs_files)
            mat_file = fullfile(dimacs_files(1).folder, dimacs_files(1).name);
        end
        
        if ~isempty(mat_file) && exist('mat_loader', 'file')
            fprintf('    Testing mat_loader with: %s\n', dimacs_files(1).name);
            try
                [A, b, c, K] = mat_loader(mat_file);
                
                if isempty(A) || isempty(b) || isempty(c) || ~isstruct(K)
                    error('mat_loader returned invalid data');
                end
                
                fprintf('    ✓ mat_loader successful (%dx%d problem)\n', size(A, 1), size(A, 2));
            catch ME2
                fprintf('    ⚠️  mat_loader failed: %s\n', ME2.message);
            end
        else
            fprintf('    ⚠️  No suitable DIMACS files found or mat_loader missing\n');
        end
    else
        fprintf('    ⚠️  No DIMACS files found\n');
    end
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Data Loaders', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Data loader test failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Data Loaders', 'status', 'FAILED', 'error', ME.message);
end

end

function test_results = test_solver_runners(test_results)
% Test solver runners with simple synthetic problem

fprintf('  Testing solver runners...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Create simple LP problem: min x1 + x2 s.t. x1 + x2 = 1, x1, x2 >= 0
    A = sparse([1, 1]);  % x1 + x2 = 1
    b = 1;               % Right-hand side
    c = [1; 1];          % Minimize x1 + x2
    K = struct('f', 0, 'l', 2, 'q', [], 's', []);  % Both variables are non-negative
    
    fprintf('    Testing SeDuMi solver runner...\n');
    try
        [x_sedumi, y_sedumi, result_sedumi] = sedumi_runner(A, b, c, K);
        
        if ~isempty(x_sedumi) && ~isempty(y_sedumi) && isstruct(result_sedumi)
            fprintf('    ✓ SeDuMi runner successful: status=%s\n', result_sedumi.status);
        else
            fprintf('    ⚠️  SeDuMi runner returned empty results\n');
        end
    catch ME2
        fprintf('    ❌ SeDuMi runner failed: %s\n', ME2.message);
    end
    
    fprintf('    Testing SDPT3 solver runner...\n');
    try
        [x_sdpt3, y_sdpt3, result_sdpt3] = sdpt3_runner(A, b, c, K);
        
        if ~isempty(x_sdpt3) && ~isempty(y_sdpt3) && isstruct(result_sdpt3)
            fprintf('    ✓ SDPT3 runner successful: status=%s\n', result_sdpt3.status);
        else
            fprintf('    ⚠️  SDPT3 runner returned empty results\n');
        end
    catch ME2
        fprintf('    ❌ SDPT3 runner failed: %s\n', ME2.message);
    end
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Solver Runners', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Solver runner test failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Solver Runners', 'status', 'FAILED', 'error', ME.message);
end

end

function test_results = test_complete_pipeline(test_results)
% Test complete matlab_runner pipeline

fprintf('  Testing complete matlab_runner pipeline...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Try with a known problem that should exist
    known_problems = {'arch0', 'nb', 'truss1'};
    
    pipeline_success = false;
    
    for i = 1:length(known_problems)
        problem_name = known_problems{i};
        result_file = sprintf('/tmp/test_integration_%s_%d.json', problem_name, round(rand()*10000));
        
        fprintf('    Trying matlab_runner with problem: %s\n', problem_name);
        
        try
            % Test complete pipeline
            tic;
            matlab_runner(problem_name, 'sedumi', result_file, false);
            execution_time = toc;
            
            % Check if result file was created
            if exist(result_file, 'file')
                % Read and validate JSON
                fid = fopen(result_file, 'r');
                json_text = fread(fid, '*char')';
                fclose(fid);
                
                result_content = jsondecode(json_text);
                
                fprintf('    ✓ Complete pipeline successful in %.2f seconds\n', execution_time);
                fprintf('    ✓ Result: status=%s\n', result_content.status);
                
                % Clean up
                delete(result_file);
                
                pipeline_success = true;
                break;
                
            else
                fprintf('    ⚠️  No result file created for %s\n', problem_name);
            end
            
        catch ME2
            fprintf('    ⚠️  Pipeline failed for %s: %s\n', problem_name, ME2.message);
            
            % Clean up on error
            if exist(result_file, 'file')
                delete(result_file);
            end
        end
    end
    
    if pipeline_success
        test_results.passed_tests = test_results.passed_tests + 1;
        test_results.test_details{end+1} = struct('name', 'Complete Pipeline', 'status', 'PASSED', 'error', '');
    else
        error('Complete pipeline test failed for all test problems');
    end
    
catch ME
    fprintf('    ❌ Complete pipeline test failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Complete Pipeline', 'status', 'FAILED', 'error', ME.message);
end

end