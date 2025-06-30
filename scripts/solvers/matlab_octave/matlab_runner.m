function matlab_runner(problem_name, solver_name, result_file, save_solutions)
% Main MATLAB orchestrator for benchmark execution
%
% Input:
%   problem_name: Name of problem from problem_registry.yaml
%   solver_name: Name of solver ('sedumi' or 'sdpt3')
%   result_file: Path to output JSON file for results
%   save_solutions: (optional) Boolean flag to save solutions to .mat file
%
% This function:
% 1. Loads problem_registry.yaml configuration
% 2. Resolves problem file path and type
% 3. Loads problem data using appropriate loader
% 4. Executes specified solver
% 5. Saves results to JSON file
% 6. Optionally saves solution vectors to .mat file

% Handle optional parameters
if nargin < 4
    save_solutions = false;
end
if nargin < 3
    save_json = false;
else
    save_json = true;
end

try
    % Add necessary paths for solvers and loaders
    addpath(genpath('scripts/'));
    
    fprintf('MATLAB Runner: Starting %s with %s\n', problem_name, solver_name);
    
    % Load problem registry configuration
    config = load_problem_registry();
    
    % Validate problem exists
    if ~isfield(config.problem_libraries, problem_name)
        error('Unknown problem: %s', problem_name);
    end
    
    problem_config = config.problem_libraries.(problem_name);
    
    % Resolve file path
    file_path = problem_config.file_path;
    file_type = problem_config.file_type;
    
    fprintf('Loading problem: %s (type: %s)\n', file_path, file_type);
    
    % Load problem data using appropriate loader
    if strcmp(file_type, 'mat')
        [A, b, c, K] = mat_loader(file_path);
    elseif strcmp(file_type, 'dat-s')
        [A, b, c, K] = dat_loader(file_path);
    else
        error('Unsupported file type: %s', file_type);
    end
    
    fprintf('Problem loaded: %d variables, %d constraints\n', size(A, 2), size(A, 1));
    
    % Execute solver and get solutions
    if strcmp(solver_name, 'sedumi')
        [x, y, result] = sedumi_runner(A, b, c, K);
    elseif strcmp(solver_name, 'sdpt3')
        [x, y, result] = sdpt3_runner(A, b, c, K);
    else
        error('Unknown solver: %s', solver_name);
    end
    
    % Calculate metrics using shared function
    if ~isempty(x) && ~isempty(y)
        result = solver_metrics_calculator(result, x, y, A, b, c, K);
    end
    
    fprintf('Solver completed with status: %s\n', result.status);
    
    % Save solution vectors if requested and solver succeeded
    if save_solutions && strcmp(result.status, 'optimal') && ~isempty(x) && ~isempty(y)
        save_solution_file(problem_name, solver_name, x, y, save_solutions);
    end
    
    if save_json
        % Convert result to JSON-compatible format (without solutions)
        json_result = convert_to_json_result(result);
        
        % Save result to JSON file
        save_json_result(json_result, result_file);
    end
    
    fprintf('MATLAB solver execution completed successfully\n');
    
catch ME
    if save_json
       % Save error result to JSON file
        error_result = create_error_result(ME, solver_name);
        save_json_result(error_result, result_file);
    end
    
    fprintf('MATLAB solver execution failed: %s\n', ME.message);
    exit(1);  % Exit with error code
end

end



function json_result = convert_to_json_result(result)
% Convert result to JSON-compatible format
json_result = struct();
json_result.solve_time = result.solve_time;
json_result.status = result.status;

% Handle optional numeric fields (convert [] to null)
if isempty(result.primal_objective) || isnan(result.primal_objective)
    json_result.primal_objective = [];
else
    json_result.primal_objective = result.primal_objective;
end

if isempty(result.dual_objective) || isnan(result.dual_objective)
    json_result.dual_objective = [];
else
    json_result.dual_objective = result.dual_objective;
end

if isempty(result.gap) || isnan(result.gap)
    json_result.gap = [];
else
    json_result.gap = result.gap;
end

if isempty(result.primal_infeasibility) || isnan(result.primal_infeasibility)
    json_result.primal_infeasibility = [];
else
    json_result.primal_infeasibility = result.primal_infeasibility;
end

if isempty(result.dual_infeasibility) || isnan(result.dual_infeasibility)
    json_result.dual_infeasibility = [];
else
    json_result.dual_infeasibility = result.dual_infeasibility;
end

if isempty(result.iterations) || isnan(result.iterations)
    json_result.iterations = [];
else
    json_result.iterations = result.iterations;
end

json_result.solver_version = result.solver_version;
json_result.solver_name = result.solver_name;
end

function error_result = create_error_result(ME, solver_name)
% Create error result structure
error_result = struct();
error_result.solve_time = 0;
error_result.status = 'error';
error_result.primal_objective = [];
error_result.dual_objective = [];
error_result.gap = [];
error_result.primal_infeasibility = [];
error_result.dual_infeasibility = [];
error_result.iterations = [];
error_result.solver_version = 'unknown';
error_result.solver_name = solver_name;
error_result.error_message = ME.message;
end

function config = load_problem_registry()
% Load problem registry YAML configuration
% This is a simplified YAML parser for the specific structure we need

config_file = 'config/problem_registry.yaml';
if ~exist(config_file, 'file')
    error('Problem registry file not found: %s', config_file);
end

% For now, use a basic implementation
% In practice, this would parse the actual YAML file
config = struct();
config.problem_libraries = struct();

% Placeholder implementation - would need actual YAML parsing
% For testing, manually define key problems
config.problem_libraries.nb = struct('file_path', 'problems/DIMACS/data/ANTENNA/nb.mat.gz', 'file_type', 'mat');
config.problem_libraries.arch0 = struct('file_path', 'problems/SDPLIB/data/arch0.dat-s', 'file_type', 'dat-s');
end