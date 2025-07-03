function matlab_runner(problem_name, solver_name, result_file, save_solutions)
    % Main MATLAB orchestrator for benchmark execution with integrated utilities
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
        
        % Load problem registry configuration using integrated YAML reader
        [problem_config, file_path] = read_problem_registry(problem_name);
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
        
        % Calculate metrics using integrated metrics calculator
        if ~isempty(x) && ~isempty(y)
            result = calculate_solver_metrics(result, x, y, A, b, c, K);
        end
        
        fprintf('Solver completed with status: %s\n', result.status);
        
        % Save solution vectors if requested and solver succeeded
        if save_solutions && strcmp(result.status, 'optimal') && ~isempty(x) && ~isempty(y)
            save_solutions_if_needed(problem_name, solver_name, x, y, save_solutions);
        end
        
        if save_json
            % Convert result to JSON-compatible format (without solutions)
            json_result = convert_to_json_result(result);
            
            % Save result to JSON file using integrated JSON saver
            save_json_safely(json_result, result_file);
        end
        
        fprintf('MATLAB solver execution completed successfully\n');
        
    catch ME
        if save_json
            % Save error result to JSON file
            error_result = create_error_result(ME, solver_name);
            save_json_safely(error_result, result_file);
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
    % Use field names that match Python SolverResult interface
    if isempty(result.primal_objective) || isnan(result.primal_objective)
        json_result.primal_objective_value = [];
    else
        json_result.primal_objective_value = result.primal_objective;
    end

    if isempty(result.dual_objective) || isnan(result.dual_objective)
        json_result.dual_objective_value = [];
    else
        json_result.dual_objective_value = result.dual_objective;
    end

    if isempty(result.gap) || isnan(result.gap)
        json_result.duality_gap = [];
    else
        json_result.duality_gap = result.gap;
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
    error_result.primal_objective_value = [];
    error_result.dual_objective_value = [];
    error_result.duality_gap = [];
    error_result.primal_infeasibility = [];
    error_result.dual_infeasibility = [];
    error_result.iterations = [];
    error_result.solver_version = 'unknown';
    error_result.solver_name = solver_name;
    error_result.error_message = ME.message;
end

% ========================================================================
% INTEGRATED UTILITY FUNCTIONS
% ========================================================================

function [problem_info, file_path] = read_problem_registry(problem_name, config_path)
    % Read problem_registry.yaml and resolve problem name to file path
    % Integrated version of matlab_yaml_reader.m

    try
        % Validate inputs
        if nargin < 1
            error('matlab_yaml_reader:InvalidInput', 'Problem name is required');
        end
        
        % Convert to char if string
        problem_name = char(problem_name);
        
        % Set default config path if not provided
        if nargin < 2 || isempty(config_path)
            config_path = 'config/problem_registry.yaml';
        end
        
        % Convert to char if string
        config_path = char(config_path);
        
        % Check if config file exists
        if ~exist(config_path, 'file')
            error('matlab_yaml_reader:FileNotFound', 'Configuration file not found: %s', config_path);
        end
        
        % Read and parse the YAML file
        fprintf('Reading configuration file %s...\n', config_path);
        
        % Initialize file handle
        fid = -1;
        
        try
            % Open file for reading
            fid = fopen(config_path, 'r');
            if fid == -1
                error('matlab_yaml_reader:FileOpenError', 'Cannot open configuration file: %s', config_path);
            end
            
            % Initialize variables
            problem_info = struct();
            file_path = '';
            found_problem = false;
            current_problem = '';
            in_problem_section = false;
            
            % Parse file line by line
            while ~feof(fid)
                line = fgets(fid);
                if ~ischar(line)
                    break;
                end
                
                % Store original line for indentation check, then trim
                original_line = line;
                line = strtrim(line);
                
                % Skip empty lines and comments
                if isempty(line) || startsWith(line, '#')
                    continue;
                end
                
                % Check for problem_libraries section
                if contains(line, 'problem_libraries:')
                    in_problem_section = true;
                    continue;
                end
                
                if ~in_problem_section
                    continue;
                end
                
                % Check for problem name (entries under problem_libraries)
                % Problem names appear with 2-space indent and end with ':'
                if startsWith(original_line, '  ') && ~startsWith(original_line, '    ') && endsWith(line, ':') && ~contains(line, '#')
                    % This is a problem name (2-space indent)
                    current_problem = strtrim(line(1:end-1));  % Remove ':' and trim
                    
                    if strcmp(current_problem, problem_name)
                        found_problem = true;
                    elseif found_problem
                        % We found our problem earlier, now we hit a new problem, so we're done
                        break;
                    end
                    continue;
                end
                
                % Parse problem attributes if we're in the target problem
                if found_problem && startsWith(original_line, '    ')
                    % This is an attribute line for the current problem (4-space indent)
                    % Format: "    attribute_name: value"
                    attr_line = line;  % line is already trimmed
                    
                    if contains(attr_line, ':')
                        % Split on first ':'
                        colon_pos = strfind(attr_line, ':');
                        if ~isempty(colon_pos)
                            attr_name = strtrim(attr_line(1:colon_pos(1)-1));
                            attr_value = strtrim(attr_line(colon_pos(1)+1:end));
                            
                            % Remove quotes if present
                            if startsWith(attr_value, '"') && endsWith(attr_value, '"')
                                attr_value = attr_value(2:end-1);
                            end
                            
                            % Parse different attribute types
                            if strcmp(attr_name, 'display_name')
                                problem_info.display_name = attr_value;
                            elseif strcmp(attr_name, 'file_path')
                                problem_info.file_path = attr_value;
                                file_path = attr_value;
                            elseif strcmp(attr_name, 'file_type')
                                problem_info.file_type = attr_value;
                            elseif strcmp(attr_name, 'library_name')
                                problem_info.library_name = attr_value;
                            elseif strcmp(attr_name, 'for_test_flag')
                                % Parse boolean
                                if strcmp(attr_value, 'true')
                                    problem_info.for_test_flag = true;
                                elseif strcmp(attr_value, 'false')
                                    problem_info.for_test_flag = false;
                                else
                                    problem_info.for_test_flag = str2double(attr_value) > 0;
                                end
                            elseif strcmp(attr_name, 'known_objective_value')
                                % Parse numeric value
                                problem_info.known_objective_value = str2double(attr_value);
                            end
                        end
                    end
                end
                
            end
            
            if fid ~= -1
                fclose(fid);
                fid = -1;  % Mark as closed
            end
            
            % Validate that we found the problem
            if ~found_problem
                error('matlab_yaml_reader:ProblemNotFound', 'Problem "%s" not found in configuration file', problem_name);
            end
            
            % Validate that we have the required fields
            if ~isfield(problem_info, 'file_path') || isempty(problem_info.file_path)
                error('matlab_yaml_reader:MissingFilePath', 'File path not found for problem "%s"', problem_name);
            end
            
            % Set defaults for missing optional fields
            if ~isfield(problem_info, 'display_name')
                problem_info.display_name = problem_name;
            end
            if ~isfield(problem_info, 'file_type')
                problem_info.file_type = 'unknown';
            end
            if ~isfield(problem_info, 'library_name')
                problem_info.library_name = 'unknown';
            end
            if ~isfield(problem_info, 'for_test_flag')
                problem_info.for_test_flag = false;
            end
            if ~isfield(problem_info, 'known_objective_value')
                problem_info.known_objective_value = NaN;
            end
            
            fprintf('Successfully loaded problem metadata for %s\n', problem_name);
            
        catch ME
            if fid ~= -1
                fclose(fid);
            end
            rethrow(ME);
        end
        
    catch ME
        % Provide informative error messages
        fprintf('Error reading configuration for problem %s: %s\n', problem_name, ME.message);
        rethrow(ME);
    end

end

function result = calculate_solver_metrics(result, x, y, A, b, c, K)
    % Calculate objective values and infeasibility measures
    % Integrated version of solver_metrics_calculator.m

    % Extract objective values
    if ~isempty(x) && ~isempty(c)
        try
            c_vec = c(:);
            x_vec = x(:);
            if length(c_vec) == length(x_vec)
                result.primal_objective = c_vec' * x_vec;
            else
                result.primal_objective = NaN;
            end
        catch
            result.primal_objective = NaN;
        end
    end

    if ~isempty(y) && ~isempty(b)
        try
            b_vec = b(:);
            y_vec = y(:);
            if length(b_vec) == length(y_vec)
                result.dual_objective = b_vec' * y_vec;
            else
                result.dual_objective = NaN;
            end
        catch
            result.dual_objective = NaN;
        end
    end

    % Calculate duality gap
    if ~isnan(result.primal_objective) && ~isnan(result.dual_objective)
        result.gap = abs(result.primal_objective - result.dual_objective);
    end

    % Calculate infeasibility measures
    if ~isempty(x) && ~isempty(y) && ~isempty(A) && ~isempty(b) && ~isempty(c)
        try
            % Primal infeasibility: ||A*x - b|| / (1 + ||b||)
            x_vec = x(:);
            b_vec = b(:);
            if size(A, 2) == length(x_vec) && size(A, 1) == length(b_vec)
                primal_residual = A * x_vec - b_vec;
                result.primal_infeasibility = norm(primal_residual) / (1 + norm(b_vec));
            end
            
            % Dual infeasibility: cone-specific calculation
            y_vec = y(:);
            c_vec = c(:);
            if size(A, 1) == length(y_vec) && size(A, 2) == length(c_vec)
                cmAty = c_vec - A' * y_vec;  % c - A'*y
                dinf2 = calculate_dual_cone_violation(cmAty, K);
                result.dual_infeasibility = sqrt(dinf2) / (1 + sum(c_vec.^2));
            end
        catch
            result.primal_infeasibility = NaN;
            result.dual_infeasibility = NaN;
        end
    end

end

function dinf2 = calculate_dual_cone_violation(cmAty, K)
    % Calculate dual cone violation matching Python logic exactly

    dinf2 = 0;
    nvar_cnt = 0;

    % Free variables
    if isfield(K, 'f') && K.f > 0
        free_vars = K.f;
        begin = nvar_cnt + 1;  % MATLAB 1-based indexing
        ending = nvar_cnt + free_vars;
        dinf2 = dinf2 + norm(cmAty(begin:ending), 2)^2;
        nvar_cnt = ending;
    end

    % Non-negative variables
    if isfield(K, 'l') && K.l > 0
        nonneg_vars = K.l;
        begin = nvar_cnt + 1;  % MATLAB 1-based indexing
        ending = nvar_cnt + nonneg_vars;
        dinf2 = dinf2 + norm(min(cmAty(begin:ending), 0), 2)^2;
        nvar_cnt = ending;
    end

    % Second-order cone constraints
    if isfield(K, 'q') && ~isempty(K.q)
        soc_cones = K.q;
        for i = 1:length(soc_cones)
            ndim = soc_cones(i);
            if ndim <= 0
                continue;
            end
            begin = nvar_cnt + 1;  % MATLAB 1-based indexing
            ending = nvar_cnt + ndim;
            dinf2 = dinf2 + norm(proj_onto_soc(-cmAty(begin:ending)), 2)^2;
            nvar_cnt = ending;
        end
    end

    % Semidefinite cone constraints
    if isfield(K, 's') && ~isempty(K.s)
        sdp_cones = K.s;
        for i = 1:length(sdp_cones)
            ndim = sdp_cones(i);
            if ndim <= 0
                continue;
            end
            begin = nvar_cnt + 1;  % MATLAB 1-based indexing
            ending = nvar_cnt + ndim * ndim;
            eigvals = eig(reshape(cmAty(begin:ending), ndim, ndim));
            neg_eigvals = min(eigvals, 0);
            dinf2 = dinf2 + sum(neg_eigvals.^2);
            nvar_cnt = ending;
        end
    end

end

function proj_z = proj_onto_soc(z)
    % Project onto second-order cone

    z0 = z(1);  % MATLAB 1-based indexing
    znorm = norm(z(2:end), 2);
    if znorm <= z0
        proj_z = z;
    elseif znorm <= -z0
        proj_z = zeros(size(z));
    else
        scale = (z0 + znorm) / 2;
        proj_z = [1; z(2:end) / znorm] * scale;
    end

end

function success = save_json_safely(result, output_file)
    % Save MATLAB solver result to JSON file with error recovery
    % Integrated version of save_json_result.m

    success = false;

    try
        % Validate inputs
        if ~isstruct(result)
            error('save_json_result:InvalidInput', 'Result must be a struct');
        end
        
        if ~(ischar(output_file) || isstring(output_file))
            error('save_json_result:InvalidInput', 'Output file must be a string');
        end
        
        fprintf('Converting result to JSON format...\n');
        
        % Convert result to JSON string
        json_str = format_result_to_json(result);
        
        fprintf('Saving to file: %s\n', output_file);
        
        % Ensure output directory exists
        [output_dir, ~, ~] = fileparts(output_file);
        if ~isempty(output_dir) && ~exist(output_dir, 'dir')
            mkdir(output_dir);
            fprintf('Created directory: %s\n', output_dir);
        end
        
        % Write JSON to file
        fid = fopen(output_file, 'w');
        if fid == -1
            error('save_json_result:FileOpenError', 'Cannot open file for writing: %s', output_file);
        end
        
        try
            fprintf(fid, '%s', json_str);
            fclose(fid);
            
            % Verify file was written correctly
            if exist(output_file, 'file')
                file_info = dir(output_file);
                if file_info.bytes > 0
                    success = true;
                    fprintf('Successfully saved JSON result (%d bytes)\n', file_info.bytes);
                else
                    error('save_json_result:EmptyFile', 'Output file is empty');
                end
            else
                error('save_json_result:FileNotCreated', 'Output file was not created');
            end
            
        catch ME
            if fid ~= -1
                fclose(fid);
            end
            rethrow(ME);
        end
        
    catch ME
        fprintf('Error saving JSON result: %s\n', ME.message);
        
        % Try to save error information to file
        try
            error_result = struct();
            error_result.solver_name = get_field_safe(result, 'solver_name', 'Unknown');
            error_result.status = 'num_error';
            error_result.solve_time = NaN;
            error_result.setup_time = NaN;
            error_result.iterations = NaN;
            error_result.primal_objective = NaN;
            error_result.dual_objective = NaN;
            error_result.gap = NaN;
            error_result.primal_infeasibility = NaN;
            error_result.dual_infeasibility = NaN;
            error_result.solver_version = get_field_safe(result, 'solver_version', 'Unknown');
            error_result.solver_options = struct();
            error_result.termination_reason = 'JSON save error';
            error_result.error_message = sprintf('Failed to save JSON result: %s', ME.message);
            error_result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            
            error_json = format_result_to_json(error_result);
            
            fid = fopen(output_file, 'w');
            if fid ~= -1
                fprintf(fid, '%s', error_json);
                fclose(fid);
                fprintf('Saved error result to file\n');
            end
            
        catch
            % Final fallback - even error saving failed
            fprintf('Failed to save error information\n');
        end
    end

end

function json_str = format_result_to_json(result)
    % Format MATLAB solver results to JSON string compatible with Python parsing
    % Integrated version of matlab_json_formatter.m

    try
        % Validate input
        if ~isstruct(result)
            error('matlab_json_formatter:InvalidInput', 'Input must be a struct');
        end
        
        % Create standardized result structure with all required fields
        standardized_result = create_standardized_result(result);
        
        % Convert to JSON-compatible format
        json_compatible = convert_to_json_compatible(standardized_result);
        
        % Encode to JSON string
        json_str = jsonencode(json_compatible);
        
        % Validate that the JSON is parseable
        validate_json_output(json_str);
        
    catch ME
        % Create error result JSON on failure
        error_result = struct();
        error_result.solver_name = get_field_safe(result, 'solver_name', 'Unknown');
        error_result.status = 'num_error';
        error_result.solve_time = NaN;
        error_result.setup_time = NaN;
        error_result.iterations = NaN;
        error_result.primal_objective = NaN;
        error_result.dual_objective = NaN;
        error_result.gap = NaN;
        error_result.primal_infeasibility = NaN;
        error_result.dual_infeasibility = NaN;
        error_result.solver_version = get_field_safe(result, 'solver_version', 'Unknown');
        error_result.solver_options = struct();
        error_result.termination_reason = 'JSON formatting error';
        error_result.error_message = sprintf('JSON formatting failed: %s', ME.message);
        
        % Convert error result and encode
        json_compatible_error = convert_to_json_compatible(error_result);
        json_str = jsonencode(json_compatible_error);
        
        fprintf('Error creating JSON: %s\n', ME.message);
    end

end

function standardized_result = create_standardized_result(result)
    % Create a standardized result structure with all required fields

    % Define required fields with default values
    required_fields = {
        'solver_name', 'Unknown';
        'status', 'unknown';
        'solve_time', NaN;
        'setup_time', NaN;
        'iterations', NaN;
        'primal_objective', NaN;
        'dual_objective', NaN;
        'gap', NaN;
        'primal_infeasibility', NaN;
        'dual_infeasibility', NaN;
        'solver_version', 'Unknown';
        'solver_options', struct();
        'termination_reason', '';
        'error_message', ''
    };

    standardized_result = struct();

    % Copy fields from input result or use defaults
    for i = 1:size(required_fields, 1)
        field_name = required_fields{i, 1};
        default_value = required_fields{i, 2};
        
        standardized_result.(field_name) = get_field_safe(result, field_name, default_value);
    end

    % Add additional metadata
    standardized_result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    standardized_result.matlab_version = version;

end

function json_compatible = convert_to_json_compatible(result)
    % Convert MATLAB result structure to JSON-compatible format

    json_compatible = struct();

    % Copy all fields and convert as needed
    field_names = fieldnames(result);
    for i = 1:length(field_names)
        field_name = field_names{i};
        field_value = result.(field_name);
        
        json_compatible.(field_name) = convert_value_to_json_compatible(field_value);
    end

end

function json_value = convert_value_to_json_compatible(value)
    % Convert individual values to JSON-compatible format

    if ischar(value) || isstring(value)
        % String values: ensure proper encoding
        json_value = char(value);
        
    elseif isnumeric(value)
        if isscalar(value)
            if isnan(value)
                % NaN becomes null in JSON
                json_value = [];  % MATLAB jsonencode converts empty to null
            elseif isinf(value)
                % Infinity handling
                if value > 0
                    json_value = 1e308;  % Large positive number
                else
                    json_value = -1e308; % Large negative number
                end
            else
                % Regular numeric value
                json_value = double(value);
            end
        else
            % Array values
            if isempty(value)
                json_value = [];
            else
                % Convert to column vector for consistency
                json_value = value(:);
                
                % Handle NaN and Inf in arrays
                nan_mask = isnan(json_value);
                if any(nan_mask)
                    % Replace NaN with null representation (empty in MATLAB)
                    json_value = json_value(~nan_mask);
                end
                
                inf_mask = isinf(json_value);
                if any(inf_mask)
                    json_value(json_value == Inf) = 1e308;
                    json_value(json_value == -Inf) = -1e308;
                end
            end
        end
        
    elseif islogical(value)
        % Boolean values
        json_value = logical(value);
        
    elseif isstruct(value)
        % Nested structures: recursively convert
        if isempty(value)
            json_value = struct();
        else
            json_value = convert_to_json_compatible(value);
        end
        
    elseif iscell(value)
        % Cell arrays: convert to regular arrays if possible
        if isempty(value)
            json_value = [];
        else
            try
                % Try to convert to numeric array
                json_value = cell2mat(value);
                json_value = convert_value_to_json_compatible(json_value);
            catch
                % Keep as cell array if conversion fails
                json_value = value;
            end
        end
        
    else
        % Unknown type: convert to string representation
        try
            json_value = char(string(value));
        catch
            json_value = 'unknown_type';
        end
    end

end

function validate_json_output(json_str)
    % Validate that the JSON string can be parsed

    try
        % Try to decode the JSON to verify it's valid
        decoded = jsondecode(json_str);
        
        % Check that essential fields exist
        required_fields = {'solver_name', 'status', 'solve_time'};
        for i = 1:length(required_fields)
            if ~isfield(decoded, required_fields{i})
                error('Missing required field: %s', required_fields{i});
            end
        end
        
    catch ME
        error('matlab_json_formatter:InvalidJSON', 'Generated JSON is invalid: %s', ME.message);
    end

end

function save_solutions_if_needed(problem_name, solver_name, x, y, save_solutions)
    % Save solution vectors to .mat file if save_solutions is true
    % Integrated version of save_solution_file.m

    if nargin < 5 || ~save_solutions
        return;  % Don't save if flag is false or not provided
    end

    try
        % Create solutions directory if it doesn't exist
        solutions_dir = 'problems/solutions';
        if ~exist(solutions_dir, 'dir')
            mkdir(solutions_dir);
        end
        
        % Generate filename
        filename = sprintf('%s_%s.mat', problem_name, solver_name);
        filepath = fullfile(solutions_dir, filename);
        
        % Prepare solution data
        solution_data = struct();
        solution_data.problem_name = problem_name;
        solution_data.solver_name = solver_name;
        solution_data.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        
        % Add solution vectors if available
        if ~isempty(x)
            solution_data.primal_solution = x;
        end
        
        if ~isempty(y)
            solution_data.dual_solution = y;
        end
        
        % Save to .mat file
        save(filepath, '-struct', 'solution_data');
        
        fprintf('Solution saved to: %s\n', filepath);
        
    catch ME
        fprintf('Warning: Failed to save solution file: %s\n', ME.message);
    end

end

function metadata = detect_versions()
    % Comprehensive version detection for MATLAB environment and solvers
    % Integrated version of matlab_version_detection.m

    fprintf('Collecting version information...\n');

    % Initialize metadata structure
    metadata = struct();
    metadata.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    try
        %% MATLAB Version Information
        matlab_info = ver('MATLAB');
        if ~isempty(matlab_info)
            metadata.matlab_version = matlab_info.Version;
            metadata.matlab_release = matlab_info.Release;
            metadata.matlab_date = matlab_info.Date;
        else
            % Fallback method
            metadata.matlab_version = version;
            metadata.matlab_release = version('-release');
            metadata.matlab_date = 'Unknown';
        end
        
        %% Platform Information
        metadata.platform = computer;
        metadata.architecture = computer('arch');
        
        %% SeDuMi Version Detection
        fprintf('  Detecting SeDuMi version...\n');
        [metadata.sedumi_available, metadata.sedumi_version] = detect_sedumi_version();
        
        %% SDPT3 Version Detection
        fprintf('  Detecting SDPT3 version...\n');
        [metadata.sdpt3_available, metadata.sdpt3_version] = detect_sdpt3_version();
        
        %% Additional Environment Information
        metadata.java_version = version('-java');
        
        % Memory information
        try
            [user_view, sys_view] = memory;
            metadata.memory_available_gb = sys_view.PhysicalMemory.Available / 1024^3;
            metadata.memory_total_gb = sys_view.PhysicalMemory.Total / 1024^3;
        catch
            metadata.memory_available_gb = NaN;
            metadata.memory_total_gb = NaN;
        end
        
        fprintf('Version detection completed successfully\n');
        
    catch ME
        fprintf('Error during detection: %s\n', ME.message);
        
        % Ensure basic fields exist even on error
        if ~isfield(metadata, 'matlab_version')
            metadata.matlab_version = 'Unknown';
        end
        if ~isfield(metadata, 'sedumi_available')
            metadata.sedumi_available = false;
            metadata.sedumi_version = 'Unknown';
        end
        if ~isfield(metadata, 'sdpt3_available')
            metadata.sdpt3_available = false;
            metadata.sdpt3_version = 'Unknown';
        end
    end

end

function [available, version_str] = detect_sedumi_version()
    % Detect SeDuMi version with multiple fallback strategies

    available = false;
    version_str = 'SeDuMi-Unknown';

    try
        % Strategy 1: Check if sedumi function exists
        if exist('sedumi', 'file') ~= 2
            fprintf('    SeDuMi function not found\n');
            return;
        end
        
        % Strategy 2: Try to call sedumi_version if it exists
        if exist('sedumi_version', 'file') == 2
            try
                version_info = sedumi_version();
                if isstruct(version_info)
                    version_str = sprintf('SeDuMi-%s', version_info.version);
                elseif ischar(version_info)
                    version_str = sprintf('SeDuMi-%s', version_info);
                else
                    version_str = 'SeDuMi-1.3 (detected via sedumi_version)';
                end
                available = true;
                fprintf('    SeDuMi version detected via sedumi_version(): %s\n', version_str);
                return;
            catch
                % sedumi_version failed, continue to next strategy
            end
        end
        
        % Strategy 3: Test basic functionality
        try
            % Create minimal test problem
            A = sparse([1, 1]);
            b = 1;
            c = [1; 1];
            K = struct('f', 0, 'l', 2, 'q', [], 's', []);
            pars = struct('fid', 0);  % Silent mode
            
            % Try to call SeDuMi
            [~, ~, info] = sedumi(A, b, c, K, pars);
            
            if isstruct(info)
                available = true;
                version_str = 'SeDuMi-1.3 (or compatible)';
                fprintf('    SeDuMi functionality confirmed: %s\n', version_str);
                return;
            end
        catch ME
            fprintf('    SeDuMi functionality test failed: %s\n', ME.message);
        end
        
        fprintf('    SeDuMi version detection failed\n');
        
    catch ME
        fprintf('    SeDuMi detection error: %s\n', ME.message);
    end

end

function [available, version_str] = detect_sdpt3_version()
    % Detect SDPT3 version with multiple fallback strategies

    available = false;
    version_str = 'SDPT3-Unknown';

    try
        % Strategy 1: Check if main SDPT3 functions exist
        if exist('sqlp', 'file') ~= 2
            fprintf('    SDPT3 function sqlp not found\n');
            return;
        end
        
        % Strategy 2: Try to get version from sqlparameters
        try
            if exist('sqlparameters', 'file') == 2
                OPTIONS = sqlparameters;
                if isstruct(OPTIONS)
                    % SDPT3 is available and working
                    available = true;
                    version_str = 'SDPT3-4.0 (or compatible)';
                    fprintf('    SDPT3 functionality confirmed: %s\n', version_str);
                    return;
                end
            end
        catch ME
            fprintf('    SDPT3 functionality test failed: %s\n', ME.message);
        end
        
        fprintf('    SDPT3 version detection failed\n');
        
    catch ME
        fprintf('    SDPT3 detection error: %s\n', ME.message);
    end

end

function field_value = get_field_safe(struct_input, field_name, default_value)
    % Safely get field value from struct with default fallback

    if isstruct(struct_input) && isfield(struct_input, field_name)
        field_value = struct_input.(field_name);
        
        % Additional type checking for specific fields
        switch field_name
            case {'solve_time', 'setup_time', 'primal_objective', 'dual_objective', 'gap', 'primal_infeasibility', 'dual_infeasibility'}
                if ~isnumeric(field_value) || ~isscalar(field_value)
                    field_value = default_value;
                end
            case 'iterations'
                if ~isnumeric(field_value) || ~isscalar(field_value)
                    field_value = default_value;
                end
            case {'solver_name', 'status', 'solver_version', 'termination_reason', 'error_message'}
                if ~(ischar(field_value) || isstring(field_value))
                    field_value = default_value;
                end
            case 'solver_options'
                if ~isstruct(field_value)
                    field_value = default_value;
                end
        end
    else
        field_value = default_value;
    end

end