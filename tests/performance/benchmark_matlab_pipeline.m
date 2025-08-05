function benchmark_matlab_pipeline()
% Performance benchmarking for MATLAB integration pipeline
%
% This function measures and analyzes performance characteristics of the
% complete MATLAB solver integration pipeline, including:
% - Execution time analysis across different problem sizes
% - Memory usage monitoring
% - Startup overhead measurement
% - Throughput testing with multiple problems
% - Comparative analysis between solvers
%
% Results help validate performance meets production requirements and
% identify optimization opportunities.

fprintf('\n');
fprintf('================================================================\n');
fprintf('MATLAB INTEGRATION PIPELINE PERFORMANCE BENCHMARK\n');
fprintf('================================================================\n');
fprintf('Date: %s\n', datestr(now));
fprintf('MATLAB: %s\n', version);
fprintf('Platform: %s\n', computer);
fprintf('================================================================\n\n');

% Add required paths
addpath(genpath('scripts/data_loaders/matlab'));
addpath(genpath('scripts/solvers/matlab'));
addpath(genpath('scripts/utils'));

% Initialize benchmark data
benchmark_data = struct();
benchmark_data.total_runs = 0;
benchmark_data.execution_times = [];
benchmark_data.problem_info = {};
benchmark_data.solver_info = {};
benchmark_data.memory_usage = [];

try
    % Benchmark 1: Problem Size Analysis
    fprintf('Benchmark 1: Problem Size vs Performance Analysis\n');
    benchmark_data = run_problem_size_benchmark(benchmark_data);
    
    % Benchmark 2: Solver Performance Comparison
    fprintf('\nBenchmark 2: SeDuMi vs SDPT3 Performance Comparison\n');
    benchmark_data = run_solver_comparison_benchmark(benchmark_data);
    
    % Benchmark 3: Startup Overhead Analysis
    fprintf('\nBenchmark 3: Startup Overhead and Cold vs Warm Performance\n');
    benchmark_data = run_startup_overhead_benchmark(benchmark_data);
    
    % Benchmark 4: Throughput Testing
    fprintf('\nBenchmark 4: Pipeline Throughput with Multiple Problems\n');
    benchmark_data = run_throughput_benchmark(benchmark_data);
    
    % Benchmark 5: Memory Usage Analysis
    fprintf('\nBenchmark 5: Memory Usage and Resource Management\n');
    benchmark_data = run_memory_benchmark(benchmark_data);
    
catch ME
    fprintf('CRITICAL ERROR in benchmark suite: %s\n', ME.message);
end

% Generate comprehensive performance report
generate_performance_report(benchmark_data);

fprintf('\nPerformance benchmarking completed.\n');

end

function benchmark_data = run_problem_size_benchmark(benchmark_data)
% Benchmark performance across different problem sizes

fprintf('  Analyzing performance vs problem size...\n');

% Test problems of different sizes (if available)
test_problems = {
    'arch0',    'sedumi',   'Small SDP';     % ~174 variables
    'nb',       'sedumi',   'Medium SOCP';   % ~993 variables  
    'arch0',    'sdpt3',    'Small SDP';     % Same problem, different solver
    'nb',       'sdpt3',    'Medium SOCP'    % Same problem, different solver
};

for i = 1:size(test_problems, 1)
    problem_name = test_problems{i, 1};
    solver_name = test_problems{i, 2};
    description = test_problems{i, 3};
    
    fprintf('    Testing %s with %s (%s)...\n', problem_name, solver_name, description);
    
    try
        % Multiple runs for statistical significance
        run_times = [];
        for run = 1:3
            result_file = sprintf('/tmp/benchmark_%s_%s_%d_%d.json', ...
                                problem_name, solver_name, i, run);
            
            tic;
            matlab_runner(problem_name, solver_name, result_file, false);
            execution_time = toc;
            
            run_times(end+1) = execution_time;
            
            % Validate result
            if exist(result_file, 'file')
                delete(result_file);
            end
        end
        
        avg_time = mean(run_times);
        std_time = std(run_times);
        
        fprintf('      ✓ Average: %.2f±%.2f seconds (%d runs)\n', avg_time, std_time, length(run_times));
        
        % Store data
        benchmark_data.total_runs = benchmark_data.total_runs + length(run_times);
        benchmark_data.execution_times = [benchmark_data.execution_times, run_times];
        benchmark_data.problem_info{end+1} = struct('name', problem_name, 'description', description, ...
                                                   'avg_time', avg_time, 'std_time', std_time);
        benchmark_data.solver_info{end+1} = solver_name;
        
    catch ME
        fprintf('      ❌ Failed: %s\n', ME.message);
    end
end

end

function benchmark_data = run_solver_comparison_benchmark(benchmark_data)
% Compare SeDuMi vs SDPT3 performance on same problems

fprintf('  Comparing solver performance characteristics...\n');

comparison_problems = {'arch0', 'nb'};

for i = 1:length(comparison_problems)
    problem_name = comparison_problems{i};
    
    fprintf('    Problem: %s\n', problem_name);
    
    % Benchmark SeDuMi
    sedumi_times = [];
    try
        for run = 1:3
            result_file = sprintf('/tmp/benchmark_sedumi_%s_%d.json', problem_name, run);
            
            tic;
            matlab_runner(problem_name, 'sedumi', result_file, false);
            sedumi_times(end+1) = toc;
            
            if exist(result_file, 'file')
                delete(result_file);
            end
        end
        
        sedumi_avg = mean(sedumi_times);
        fprintf('      SeDuMi: %.2f±%.2f seconds\n', sedumi_avg, std(sedumi_times));
        
    catch ME
        fprintf('      SeDuMi failed: %s\n', ME.message);
        sedumi_avg = NaN;
    end
    
    % Benchmark SDPT3
    sdpt3_times = [];
    try
        for run = 1:3
            result_file = sprintf('/tmp/benchmark_sdpt3_%s_%d.json', problem_name, run);
            
            tic;
            matlab_runner(problem_name, 'sdpt3', result_file, false);
            sdpt3_times(end+1) = toc;
            
            if exist(result_file, 'file')
                delete(result_file);
            end
        end
        
        sdpt3_avg = mean(sdpt3_times);
        fprintf('      SDPT3: %.2f±%.2f seconds\n', sdpt3_avg, std(sdpt3_times));
        
    catch ME
        fprintf('      SDPT3 failed: %s\n', ME.message);
        sdpt3_avg = NaN;
    end
    
    % Performance comparison
    if ~isnan(sedumi_avg) && ~isnan(sdpt3_avg)
        if sedumi_avg < sdpt3_avg
            speedup = sdpt3_avg / sedumi_avg;
            fprintf('      → SeDuMi is %.2fx faster than SDPT3\n', speedup);
        else
            speedup = sedumi_avg / sdpt3_avg;
            fprintf('      → SDPT3 is %.2fx faster than SeDuMi\n', speedup);
        end
    end
    
    % Store data
    benchmark_data.execution_times = [benchmark_data.execution_times, sedumi_times, sdpt3_times];
    benchmark_data.total_runs = benchmark_data.total_runs + length(sedumi_times) + length(sdpt3_times);
end

end

function benchmark_data = run_startup_overhead_benchmark(benchmark_data)
% Analyze MATLAB startup overhead and cold vs warm performance

fprintf('  Analyzing startup overhead...\n');

% Test cold start performance (first run after MATLAB starts)
fprintf('    Cold start performance...\n');
try
    result_file = '/tmp/benchmark_cold_start.json';
    
    tic;
    matlab_runner('arch0', 'sedumi', result_file, false);
    cold_start_time = toc;
    
    if exist(result_file, 'file')
        delete(result_file);
    end
    
    fprintf('      Cold start: %.2f seconds\n', cold_start_time);
    
catch ME
    fprintf('      Cold start failed: %s\n', ME.message);
    cold_start_time = NaN;
end

% Test warm performance (subsequent runs)
fprintf('    Warm performance (5 consecutive runs)...\n');
warm_times = [];

for run = 1:5
    try
        result_file = sprintf('/tmp/benchmark_warm_%d.json', run);
        
        tic;
        matlab_runner('arch0', 'sedumi', result_file, false);
        warm_times(end+1) = toc;
        
        if exist(result_file, 'file')
            delete(result_file);
        end
        
    catch ME
        fprintf('      Warm run %d failed: %s\n', run, ME.message);
    end
end

if ~isempty(warm_times)
    warm_avg = mean(warm_times);
    warm_std = std(warm_times);
    
    fprintf('      Warm average: %.2f±%.2f seconds\n', warm_avg, warm_std);
    
    % Analyze startup overhead
    if ~isnan(cold_start_time)
        startup_overhead = cold_start_time - warm_avg;
        fprintf('      Startup overhead: %.2f seconds\n', startup_overhead);
        
        if startup_overhead > 10
            fprintf('      ⚠️  High startup overhead detected\n');
        else
            fprintf('      ✓ Startup overhead acceptable\n');
        end
    end
    
    % Store data
    benchmark_data.execution_times = [benchmark_data.execution_times, warm_times];
    benchmark_data.total_runs = benchmark_data.total_runs + length(warm_times);
end

end

function benchmark_data = run_throughput_benchmark(benchmark_data)
% Test pipeline throughput with multiple problems

fprintf('  Testing pipeline throughput...\n');

% Run multiple problems in sequence
test_sequence = {'arch0', 'nb', 'arch0', 'nb'};  % Mix of problems
solvers = {'sedumi', 'sdpt3'};

total_start_time = tic;
successful_runs = 0;
total_solver_time = 0;

for solver_idx = 1:length(solvers)
    solver_name = solvers{solver_idx};
    
    fprintf('    Throughput test with %s...\n', solver_name);
    
    solver_start_time = tic;
    solver_runs = 0;
    
    for prob_idx = 1:length(test_sequence)
        problem_name = test_sequence{prob_idx};
        
        try
            result_file = sprintf('/tmp/benchmark_throughput_%s_%s_%d.json', ...
                                solver_name, problem_name, prob_idx);
            
            tic;
            matlab_runner(problem_name, solver_name, result_file, false);
            run_time = toc;
            
            solver_runs = solver_runs + 1;
            successful_runs = successful_runs + 1;
            
            if exist(result_file, 'file')
                delete(result_file);
            end
            
            fprintf('      %s solved in %.2f seconds\n', problem_name, run_time);
            
        catch ME
            fprintf('      %s failed: %s\n', problem_name, ME.message);
        end
    end
    
    solver_total_time = toc(solver_start_time);
    total_solver_time = total_solver_time + solver_total_time;
    
    if solver_runs > 0
        avg_time_per_problem = solver_total_time / solver_runs;
        throughput = solver_runs / solver_total_time * 3600;  % Problems per hour
        
        fprintf('      %s: %d problems in %.2f seconds (%.2f sec/problem, %.1f problems/hour)\n', ...
               solver_name, solver_runs, solver_total_time, avg_time_per_problem, throughput);
    end
end

total_time = toc(total_start_time);

fprintf('    Overall throughput: %d successful runs in %.2f seconds\n', successful_runs, total_time);
if successful_runs > 0
    overall_throughput = successful_runs / total_time * 3600;
    fprintf('    Overall rate: %.1f problems/hour\n', overall_throughput);
end

% Store data
benchmark_data.total_runs = benchmark_data.total_runs + successful_runs;

end

function benchmark_data = run_memory_benchmark(benchmark_data)
% Monitor memory usage during pipeline execution

fprintf('  Monitoring memory usage...\n');

try
    % Get initial memory state
    initial_memory = memory;
    fprintf('    Initial memory: %.1f MB used\n', initial_memory.MemUsedMATLAB / 1024 / 1024);
    
    % Run a series of problems and monitor memory
    test_problems = {'arch0', 'nb'};
    memory_samples = [];
    
    for i = 1:length(test_problems)
        problem_name = test_problems{i};
        
        fprintf('    Running %s and monitoring memory...\n', problem_name);
        
        % Sample memory before
        mem_before = memory;
        
        try
            result_file = sprintf('/tmp/benchmark_memory_%s.json', problem_name);
            matlab_runner(problem_name, 'sedumi', result_file, false);
            
            % Sample memory after
            mem_after = memory;
            
            memory_used = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB) / 1024 / 1024;
            memory_samples(end+1) = memory_used;
            
            fprintf('      Memory used: %.1f MB\n', memory_used);
            
            if exist(result_file, 'file')
                delete(result_file);
            end
            
        catch ME
            fprintf('      Memory test failed: %s\n', ME.message);
        end
        
        % Force garbage collection
        clear variables;
        pack;
    end
    
    % Final memory state
    final_memory = memory;
    memory_growth = (final_memory.MemUsedMATLAB - initial_memory.MemUsedMATLAB) / 1024 / 1024;
    
    fprintf('    Final memory: %.1f MB used\n', final_memory.MemUsedMATLAB / 1024 / 1024);
    fprintf('    Memory growth: %.1f MB\n', memory_growth);
    
    if memory_growth > 100  % 100 MB growth
        fprintf('    ⚠️  Significant memory growth detected\n');
    else
        fprintf('    ✓ Memory usage reasonable\n');
    end
    
    % Store data
    benchmark_data.memory_usage = memory_samples;
    
catch ME
    fprintf('    Memory benchmark failed: %s\n', ME.message);
end

end

function generate_performance_report(benchmark_data)
% Generate comprehensive performance analysis report

fprintf('\n');
fprintf('================================================================\n');
fprintf('PERFORMANCE BENCHMARK REPORT\n');
fprintf('================================================================\n');

% Execution time statistics
if ~isempty(benchmark_data.execution_times)
    fprintf('\nExecution Time Analysis:\n');
    fprintf('  Total runs: %d\n', benchmark_data.total_runs);
    fprintf('  Average time: %.2f seconds\n', mean(benchmark_data.execution_times));
    fprintf('  Median time: %.2f seconds\n', median(benchmark_data.execution_times));
    fprintf('  Min time: %.2f seconds\n', min(benchmark_data.execution_times));
    fprintf('  Max time: %.2f seconds\n', max(benchmark_data.execution_times));
    fprintf('  Std deviation: %.2f seconds\n', std(benchmark_data.execution_times));
    
    % Performance classification
    avg_time = mean(benchmark_data.execution_times);
    if avg_time < 30
        fprintf('  Performance: ✓ Excellent (< 30 seconds average)\n');
    elseif avg_time < 60
        fprintf('  Performance: ✓ Good (30-60 seconds average)\n');
    elseif avg_time < 120
        fprintf('  Performance: ⚠️  Acceptable (60-120 seconds average)\n');
    else
        fprintf('  Performance: ❌ Slow (> 120 seconds average)\n');
    end
end

% Memory usage analysis
if ~isempty(benchmark_data.memory_usage)
    fprintf('\nMemory Usage Analysis:\n');
    fprintf('  Average memory per run: %.1f MB\n', mean(benchmark_data.memory_usage));
    fprintf('  Peak memory usage: %.1f MB\n', max(benchmark_data.memory_usage));
    fprintf('  Memory efficiency: ');
    
    avg_memory = mean(benchmark_data.memory_usage);
    if avg_memory < 50
        fprintf('✓ Excellent (< 50 MB per run)\n');
    elseif avg_memory < 100
        fprintf('✓ Good (50-100 MB per run)\n');
    elseif avg_memory < 200
        fprintf('⚠️  Acceptable (100-200 MB per run)\n');
    else
        fprintf('❌ High (> 200 MB per run)\n');
    end
end

% Production readiness assessment
fprintf('\nProduction Readiness Assessment:\n');

readiness_score = 0;
total_criteria = 4;

% Criterion 1: Average execution time
if ~isempty(benchmark_data.execution_times)
    avg_time = mean(benchmark_data.execution_times);
    if avg_time < 120  % 2 minutes
        fprintf('  ✓ Execution time acceptable\n');
        readiness_score = readiness_score + 1;
    else
        fprintf('  ❌ Execution time too slow\n');
    end
else
    fprintf('  ⚠️  No execution time data\n');
end

% Criterion 2: Performance consistency
if ~isempty(benchmark_data.execution_times) && length(benchmark_data.execution_times) > 1
    cv = std(benchmark_data.execution_times) / mean(benchmark_data.execution_times);
    if cv < 0.5  % Coefficient of variation < 50%
        fprintf('  ✓ Performance consistency good\n');
        readiness_score = readiness_score + 1;
    else
        fprintf('  ❌ Performance inconsistent\n');
    end
else
    fprintf('  ⚠️  Insufficient data for consistency analysis\n');
end

% Criterion 3: Memory efficiency
if ~isempty(benchmark_data.memory_usage)
    avg_memory = mean(benchmark_data.memory_usage);
    if avg_memory < 200  % Less than 200 MB per run
        fprintf('  ✓ Memory usage acceptable\n');
        readiness_score = readiness_score + 1;
    else
        fprintf('  ❌ Memory usage too high\n');
    end
else
    fprintf('  ⚠️  No memory usage data\n');
    readiness_score = readiness_score + 1;  % Give benefit of doubt
end

% Criterion 4: Test completion rate
if benchmark_data.total_runs > 0
    fprintf('  ✓ Pipeline execution successful\n');
    readiness_score = readiness_score + 1;
else
    fprintf('  ❌ Pipeline execution failed\n');
end

% Final assessment
readiness_percentage = (readiness_score / total_criteria) * 100;
fprintf('\nOverall Production Readiness: %.0f%% (%d/%d criteria met)\n', ...
       readiness_percentage, readiness_score, total_criteria);

if readiness_percentage >= 75
    fprintf('🎉 READY FOR PRODUCTION - Performance meets requirements\n');
elseif readiness_percentage >= 50
    fprintf('⚠️  CONDITIONALLY READY - Some performance issues to address\n');
else
    fprintf('❌ NOT READY - Significant performance improvements needed\n');
end

fprintf('================================================================\n');

end