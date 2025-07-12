function benchmark_matlab_loaders()
% Performance benchmarking for MATLAB data loaders
%
% This function measures the loading performance of different problem sizes
% and file types to ensure acceptable performance for production use.

fprintf('=== MATLAB Data Loader Performance Benchmark ===\n\n');

% Add required paths
addpath(fullfile(pwd, 'scripts/utils/'));
addpath(fullfile(pwd, 'scripts/data_loaders/matlab_octave/'));

% Performance test cases: [problem_name, expected_type]
performance_cases = {
    'nb', 'Small DIMACS';
    'nb_L2', 'Medium DIMACS';
    'bm1', 'Large DIMACS';
    'arch0', 'Medium SDPLIB';
    'control1', 'Small SDPLIB'
};

fprintf('Performance benchmarks (3 runs each):\n\n');

total_time = 0;
problem_count = 0;

for i = 1:size(performance_cases, 1)
    problem_name = performance_cases{i, 1};
    problem_type = performance_cases{i, 2};
    
    try
        % Get problem information
        [info, file_path] = matlab_yaml_reader(problem_name);
        
        % Measure loading time (3 runs)
        times = zeros(3, 1);
        
        for run = 1:3
            tic;
            if strcmp(info.file_type, 'mat')
                [A, b, c, K] = mat_loader(file_path);
            elseif strcmp(info.file_type, 'dat-s')
                [A, b, c, K] = dat_loader(file_path);
            end
            times(run) = toc;
        end
        
        % Calculate statistics
        mean_time = mean(times);
        std_time = std(times);
        min_time = min(times);
        
        % Get problem size information
        [m, n] = size(A);
        density = 100 * nnz(A) / (m * n);
        
        fprintf('%s (%s):\n', problem_name, problem_type);
        fprintf('  Size: %dx%d (%.2f%% dense)\n', m, n, density);
        fprintf('  Loading time: %.3f ± %.3f sec (min: %.3f)\n', mean_time, std_time, min_time);
        fprintf('  Performance: %.0f vars/sec\n\n', n / mean_time);
        
        total_time = total_time + mean_time;
        problem_count = problem_count + 1;
        
    catch ME
        fprintf('%s (%s): ❌ Error: %s\n\n', problem_name, problem_type, ME.message);
    end
end

fprintf('Overall Performance Summary:\n');
fprintf('  Total problems tested: %d\n', problem_count);
fprintf('  Average loading time: %.3f sec\n', total_time / problem_count);
fprintf('  Total benchmark time: %.3f sec\n', total_time);

% Performance acceptance criteria
if total_time / problem_count < 2.0  % Average < 2 seconds per problem
    fprintf('  Performance: ✅ ACCEPTABLE\n');
else
    fprintf('  Performance: ⚠️  SLOW (consider optimization)\n');
end

fprintf('\nBenchmark completed successfully!\n');

end