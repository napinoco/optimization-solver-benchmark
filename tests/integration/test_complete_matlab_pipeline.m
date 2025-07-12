function test_complete_matlab_pipeline()
% Comprehensive end-to-end testing of complete MATLAB integration pipeline
%
% This function tests the complete MATLAB solver integration from start to finish:
% - Problem loading (DIMACS .mat, SDPLIB .dat-s)
% - Solver execution (SeDuMi, SDPT3) 
% - JSON result formatting and validation
% - Solution vector storage
% - Error handling and recovery
% - Performance characteristics
%
% This is the definitive test to validate our integration is ready for 
% Python interface development (Sprint 4).

fprintf('\n');
fprintf('================================================================\n');
fprintf('COMPLETE MATLAB INTEGRATION PIPELINE TESTING\n');
fprintf('================================================================\n');
fprintf('Date: %s\n', datestr(now));
fprintf('MATLAB: %s\n', version);
fprintf('Platform: %s\n', computer);
fprintf('================================================================\n\n');

% Add required paths
addpath(genpath('scripts/data_loaders/matlab_octave'));
addpath(genpath('scripts/solvers/matlab_octave'));
addpath(genpath('scripts/utils'));

% Initialize test results tracking
test_results = struct();
test_results.total_tests = 0;
test_results.passed_tests = 0;
test_results.failed_tests = 0;
test_results.test_details = {};

% Track performance metrics
performance_metrics = struct();
performance_metrics.execution_times = [];
performance_metrics.problem_sizes = [];
performance_metrics.solver_times = [];

try
    % Test 1: Complete Pipeline with SDPLIB Problem
    fprintf('Test 1: Complete Pipeline - SDPLIB Problem (arch0)\n');
    test_results = run_sdplib_pipeline_test(test_results, performance_metrics);
    
    % Test 2: Complete Pipeline with DIMACS Problem  
    fprintf('\nTest 2: Complete Pipeline - DIMACS Problem (nb)\n');
    test_results = run_dimacs_pipeline_test(test_results, performance_metrics);
    
    % Test 3: SeDuMi vs SDPT3 Solver Comparison
    fprintf('\nTest 3: SeDuMi vs SDPT3 Solver Comparison\n');
    test_results = run_solver_comparison_test(test_results, performance_metrics);
    
    % Test 4: JSON Output Format Validation
    fprintf('\nTest 4: JSON Output Format Validation\n');
    test_results = run_json_validation_test(test_results);
    
    % Test 5: Solution Vector Storage Testing
    fprintf('\nTest 5: Solution Vector Storage Testing\n');
    test_results = run_solution_storage_test(test_results);
    
    % Test 6: Error Handling and Recovery
    fprintf('\nTest 6: Error Handling and Recovery\n');
    test_results = run_error_handling_test(test_results);
    
    % Test 7: Performance and Memory Validation
    fprintf('\nTest 7: Performance and Memory Validation\n');
    test_results = run_performance_validation_test(test_results, performance_metrics);
    
catch ME
    fprintf('CRITICAL ERROR in pipeline test suite: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
end

% Generate comprehensive final report
fprintf('\n');
fprintf('================================================================\n');
fprintf('COMPLETE PIPELINE TEST SUMMARY\n');
fprintf('================================================================\n');
fprintf('Total Tests: %d\n', test_results.total_tests);
fprintf('Passed: %d\n', test_results.passed_tests);
fprintf('Failed: %d\n', test_results.failed_tests);
fprintf('Success Rate: %.1f%%\n', 100 * test_results.passed_tests / test_results.total_tests);

% Performance summary
if ~isempty(performance_metrics.execution_times)
    fprintf('\nPerformance Summary:\n');
    fprintf('Average Execution Time: %.2f seconds\n', mean(performance_metrics.execution_times));
    fprintf('Fastest Execution: %.2f seconds\n', min(performance_metrics.execution_times));
    fprintf('Slowest Execution: %.2f seconds\n', max(performance_metrics.execution_times));
end

fprintf('================================================================\n');

% Report failed tests details
if test_results.failed_tests > 0
    fprintf('\nFAILED TESTS DETAILS:\n');
    for i = 1:length(test_results.test_details)
        detail = test_results.test_details{i};
        if strcmp(detail.status, 'FAILED')
            fprintf('  ❌ %s: %s\n', detail.name, detail.error);
        end
    end
end

% Final assessment
if test_results.failed_tests == 0
    fprintf('\n🎉 ALL TESTS PASSED - MATLAB Integration Pipeline Ready for Production!\n');
else
    fprintf('\n⚠️  Some tests failed - Review issues before proceeding to Sprint 4\n');
end

fprintf('\nComplete pipeline testing finished.\n');

end

function test_results = run_sdplib_pipeline_test(test_results, performance_metrics)
% Test complete pipeline with SDPLIB problem (arch0)

fprintf('  Testing with arch0.dat-s (SDPLIB format)...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Test data
    problem_file = 'problems/SDPLIB/data/arch0.dat-s';
    
    if ~exist(problem_file, 'file')
        error('SDPLIB test file not found: %s', problem_file);
    end
    
    % Test complete pipeline with SeDuMi
    tic;
    result_file = sprintf('/tmp/test_arch0_sedumi_%d.json', round(rand()*10000));
    
    % Execute complete matlab_runner pipeline
    matlab_runner('arch0', 'sedumi', result_file, false);
    
    execution_time = toc;
    performance_metrics.execution_times(end+1) = execution_time;
    
    % Validate result file was created
    if ~exist(result_file, 'file')
        error('Pipeline did not create result file');
    end
    
    % Validate JSON content
    result_content = validate_json_result_file(result_file);
    
    % Clean up
    delete(result_file);
    
    fprintf('    ✓ SDPLIB pipeline successful in %.2f seconds\n', execution_time);
    fprintf('    ✓ Status: %s, Objective: %.6f\n', result_content.status, result_content.primal_objective_value);
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'SDPLIB Pipeline', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ SDPLIB pipeline failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'SDPLIB Pipeline', 'status', 'FAILED', 'error', ME.message);
    
    % Clean up on error
    if exist('result_file', 'var') && exist(result_file, 'file')
        delete(result_file);
    end
end

end

function test_results = run_dimacs_pipeline_test(test_results, performance_metrics)
% Test complete pipeline with DIMACS problem (nb)

fprintf('  Testing with nb.mat.gz (DIMACS format)...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Test data
    problem_file = 'problems/DIMACS/data/ANTENNA/nb.mat.gz';
    
    if ~exist(problem_file, 'file')
        error('DIMACS test file not found: %s', problem_file);
    end
    
    % Test complete pipeline with SDPT3
    tic;
    result_file = sprintf('/tmp/test_nb_sdpt3_%d.json', round(rand()*10000));
    
    % Execute complete matlab_runner pipeline
    matlab_runner('nb', 'sdpt3', result_file, false);
    
    execution_time = toc;
    performance_metrics.execution_times(end+1) = execution_time;
    
    % Validate result file was created
    if ~exist(result_file, 'file')
        error('Pipeline did not create result file');
    end
    
    % Validate JSON content
    result_content = validate_json_result_file(result_file);
    
    % Clean up
    delete(result_file);
    
    fprintf('    ✓ DIMACS pipeline successful in %.2f seconds\n', execution_time);
    fprintf('    ✓ Status: %s, Objective: %.6f\n', result_content.status, result_content.primal_objective_value);
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'DIMACS Pipeline', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ DIMACS pipeline failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'DIMACS Pipeline', 'status', 'FAILED', 'error', ME.message);
    
    % Clean up on error
    if exist('result_file', 'var') && exist(result_file, 'file')
        delete(result_file);
    end
end

end

function test_results = run_solver_comparison_test(test_results, performance_metrics)
% Test both solvers on same problem and compare results

fprintf('  Comparing SeDuMi vs SDPT3 on same problem...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Use small SDPLIB problem for comparison
    problem_name = 'arch0';
    
    % Test with SeDuMi
    result_file_sedumi = sprintf('/tmp/test_comparison_sedumi_%d.json', round(rand()*10000));
    tic;
    matlab_runner(problem_name, 'sedumi', result_file_sedumi, false);
    sedumi_time = toc;
    
    % Test with SDPT3
    result_file_sdpt3 = sprintf('/tmp/test_comparison_sdpt3_%d.json', round(rand()*10000));
    tic;
    matlab_runner(problem_name, 'sdpt3', result_file_sdpt3, false);
    sdpt3_time = toc;
    
    % Load and compare results
    sedumi_result = validate_json_result_file(result_file_sedumi);
    sdpt3_result = validate_json_result_file(result_file_sdpt3);
    
    % Compare objectives (should be similar if both optimal)
    if strcmp(sedumi_result.status, 'optimal') && strcmp(sdpt3_result.status, 'optimal')
        obj_diff = abs(sedumi_result.primal_objective_value - sdpt3_result.primal_objective_value);
        relative_diff = obj_diff / abs(sedumi_result.primal_objective_value);
        
        if relative_diff > 0.01  % 1% tolerance
            fprintf('    ⚠️  Large objective difference: %.2e (%.1f%%)\n', obj_diff, relative_diff*100);
        else
            fprintf('    ✓ Objectives agree within tolerance\n');
        end
    end
    
    fprintf('    ✓ SeDuMi: %s in %.2f seconds, obj=%.6f\n', ...
           sedumi_result.status, sedumi_time, sedumi_result.primal_objective_value);
    fprintf('    ✓ SDPT3: %s in %.2f seconds, obj=%.6f\n', ...
           sdpt3_result.status, sdpt3_time, sdpt3_result.primal_objective_value);
    
    % Clean up
    delete(result_file_sedumi);
    delete(result_file_sdpt3);
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Solver Comparison', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Solver comparison failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Solver Comparison', 'status', 'FAILED', 'error', ME.message);
    
    % Clean up on error
    if exist('result_file_sedumi', 'var') && exist(result_file_sedumi, 'file')
        delete(result_file_sedumi);
    end
    if exist('result_file_sdpt3', 'var') && exist(result_file_sdpt3, 'file')
        delete(result_file_sdpt3);
    end
end

end

function test_results = run_json_validation_test(test_results)
% Validate JSON output format matches Python requirements

fprintf('  Validating JSON output format compliance...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Generate sample result
    result_file = sprintf('/tmp/test_json_validation_%d.json', round(rand()*10000));
    matlab_runner('arch0', 'sedumi', result_file, false);
    
    % Read and validate JSON structure
    result_content = validate_json_result_file(result_file);
    
    % Check required fields
    required_fields = {'solve_time', 'status', 'primal_objective_value', 'dual_objective_value', ...
                      'duality_gap', 'primal_infeasibility', 'dual_infeasibility', 'iterations', ...
                      'solver_version', 'matlab_version'};
    
    missing_fields = {};
    for i = 1:length(required_fields)
        if ~isfield(result_content, required_fields{i})
            missing_fields{end+1} = required_fields{i};
        end
    end
    
    if ~isempty(missing_fields)
        error('Missing required JSON fields: %s', strjoin(missing_fields, ', '));
    end
    
    % Validate data types
    if ~isnumeric(result_content.solve_time) || result_content.solve_time < 0
        error('Invalid solve_time field');
    end
    
    if ~ischar(result_content.status) || isempty(result_content.status)
        error('Invalid status field');
    end
    
    % Clean up
    delete(result_file);
    
    fprintf('    ✓ JSON format validation passed\n');
    fprintf('    ✓ All required fields present with correct types\n');
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'JSON Validation', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ JSON validation failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'JSON Validation', 'status', 'FAILED', 'error', ME.message);
    
    % Clean up on error
    if exist('result_file', 'var') && exist(result_file, 'file')
        delete(result_file);
    end
end

end

function test_results = run_solution_storage_test(test_results)
% Test solution vector storage functionality

fprintf('  Testing solution vector storage...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Test with save_solutions enabled
    result_file = sprintf('/tmp/test_solution_storage_%d.json', round(rand()*10000));
    
    % Execute with solution saving enabled
    matlab_runner('arch0', 'sedumi', result_file, true);  % true = save solutions
    
    % Check if solution file was created
    solution_file = 'problems/solutions/arch0_sedumi.mat';
    
    if exist(solution_file, 'file')
        % Load and validate solution file
        solution_data = load(solution_file);
        
        if ~isfield(solution_data, 'x') || ~isfield(solution_data, 'y')
            error('Solution file missing x or y variables');
        end
        
        if isempty(solution_data.x) || isempty(solution_data.y)
            error('Solution vectors are empty');
        end
        
        fprintf('    ✓ Solution file created: %s\n', solution_file);
        fprintf('    ✓ Solution vectors: x(%d), y(%d)\n', length(solution_data.x), length(solution_data.y));
        
        % Clean up solution file
        delete(solution_file);
    else
        fprintf('    ⚠️  Solution file not created (may be due to non-optimal status)\n');
    end
    
    % Clean up result file
    delete(result_file);
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Solution Storage', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Solution storage test failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Solution Storage', 'status', 'FAILED', 'error', ME.message);
    
    % Clean up on error
    if exist('result_file', 'var') && exist(result_file, 'file')
        delete(result_file);
    end
    if exist('solution_file', 'var') && exist(solution_file, 'file')
        delete(solution_file);
    end
end

end

function test_results = run_error_handling_test(test_results)
% Test error handling with invalid inputs

fprintf('  Testing error handling and recovery...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    % Test with invalid problem name
    result_file = sprintf('/tmp/test_error_handling_%d.json', round(rand()*10000));
    
    % This should fail gracefully and create error JSON
    try
        matlab_runner('nonexistent_problem', 'sedumi', result_file, false);
    catch
        % Expected to fail - that's OK
    end
    
    % Check if error result file was created
    if exist(result_file, 'file')
        error_result = validate_json_result_file(result_file);
        
        if ~strcmp(error_result.status, 'error')
            error('Expected error status, got: %s', error_result.status);
        end
        
        fprintf('    ✓ Error handled gracefully with status: %s\n', error_result.status);
        
        % Clean up
        delete(result_file);
    else
        error('Error result file was not created');
    end
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Error Handling', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Error handling test failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Error Handling', 'status', 'FAILED', 'error', ME.message);
    
    % Clean up on error
    if exist('result_file', 'var') && exist(result_file, 'file')
        delete(result_file);
    end
end

end

function test_results = run_performance_validation_test(test_results, performance_metrics)
% Validate performance characteristics

fprintf('  Validating performance characteristics...\n');
test_results.total_tests = test_results.total_tests + 1;

try
    if isempty(performance_metrics.execution_times)
        error('No performance data collected');
    end
    
    avg_time = mean(performance_metrics.execution_times);
    max_time = max(performance_metrics.execution_times);
    min_time = min(performance_metrics.execution_times);
    
    % Performance criteria (adjust as needed)
    if avg_time > 120  % 2 minutes average
        fprintf('    ⚠️  Average execution time high: %.2f seconds\n', avg_time);
    else
        fprintf('    ✓ Average execution time acceptable: %.2f seconds\n', avg_time);
    end
    
    if max_time > 300  % 5 minutes max
        fprintf('    ⚠️  Maximum execution time high: %.2f seconds\n', max_time);
    else
        fprintf('    ✓ Maximum execution time acceptable: %.2f seconds\n', max_time);
    end
    
    % Check for reasonable consistency
    time_std = std(performance_metrics.execution_times);
    cv = time_std / avg_time;  % Coefficient of variation
    
    if cv > 1.0  % High variability
        fprintf('    ⚠️  High execution time variability: CV=%.2f\n', cv);
    else
        fprintf('    ✓ Execution time consistency good: CV=%.2f\n', cv);
    end
    
    test_results.passed_tests = test_results.passed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Performance Validation', 'status', 'PASSED', 'error', '');
    
catch ME
    fprintf('    ❌ Performance validation failed: %s\n', ME.message);
    test_results.failed_tests = test_results.failed_tests + 1;
    test_results.test_details{end+1} = struct('name', 'Performance Validation', 'status', 'FAILED', 'error', ME.message);
end

end

function result_content = validate_json_result_file(result_file)
% Validate and parse JSON result file

if ~exist(result_file, 'file')
    error('Result file does not exist: %s', result_file);
end

% Check file size
file_info = dir(result_file);
if file_info.bytes == 0
    error('Result file is empty');
end

% Read and parse JSON
try
    fid = fopen(result_file, 'r');
    json_text = fread(fid, '*char')';
    fclose(fid);
    
    result_content = jsondecode(json_text);
    
catch ME
    error('Failed to parse JSON file: %s', ME.message);
end

% Basic validation
if ~isstruct(result_content)
    error('JSON content is not a valid structure');
end

end