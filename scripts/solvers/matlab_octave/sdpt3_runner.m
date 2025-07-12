function [x, y, result] = sdpt3_runner(A, b, c, K, options)
    % Execute SDPT3 solver and return solutions with result structure
    %
    % This function runs the SDPT3 optimization solver with minimal configuration
    % for fair benchmarking. It converts SeDuMi format to SDPT3 format internally
    % and returns the solutions with a standardized result structure.
    %
    % Input:
    %   A: Constraint matrix (sparse, m x n) in SeDuMi format
    %   b: Right-hand side vector (m x 1) in SeDuMi format
    %   c: Objective vector (n x 1) in SeDuMi format
    %   K: Cone structure (struct with fields K.f, K.l, K.q, K.s) in SeDuMi format
    %   options: (optional) Solver options struct
    %
    % Output:
    %   x: Primal solution vector (in SeDuMi format)
    %   y: Dual solution vector
    %   result: Standardized result structure

    % Initialize empty outputs
    x = [];
    y = [];
    result = struct();

    try
        % Validate inputs
        if nargin < 4
            error('sdpt3_runner:InvalidInput', 'Insufficient input arguments. Need A, b, c, K');
        end
        
        % Validate problem dimensions
        [m, n] = size(A);
        if length(b) ~= m
            error('sdpt3_runner:DimensionMismatch', 'Dimension mismatch: length(b) must equal size(A,1)');
        end
        if length(c) ~= n
            error('sdpt3_runner:DimensionMismatch', 'Dimension mismatch: length(c) must equal size(A,2)');
        end
        
        % Set up minimal SDPT3 options for fair benchmarking
        if nargin < 5 || isempty(options)
            options = struct();
        end
        
        % Default SDPT3 options - minimal configuration for fair comparison
        default_options = struct();
        default_options.printlevel = 0;     % No output (silent mode)
        
        % Merge with user options (user options override defaults)
        OPTIONS = merge_options(default_options, options);
        
        % Convert SeDuMi format to SDPT3 format
        [blk, At, C, b_sdpt3, perm] = read_sedumi(A, b, c, K);
        
        % Call SDPT3 solver
        fprintf('SDPT3: Starting solve with %d variables, %d constraints\n', n, m);
        
        % Measure solve time
        solve_start_time = tic;
        [obj, X, y, Z, info, runhist] = sdpt3(blk, At, C, b_sdpt3, OPTIONS);
        solve_time = toc(solve_start_time);
        
        % Convert SDPT3 solution back to SeDuMi format
        if ~isempty(X) && iscell(X)
            try
                % Use SDPT3's built-in conversion function
                [x, ~, ~] = SDPT3soln_SEDUMIsoln(blk, X, y, Z, perm);
            catch
                x = [];
            end
        end
        
        fprintf('SDPT3: Completed in %.3f seconds\n', solve_time);
        
        % Create result structure from SDPT3 info
        result = create_sdpt3_result(info);
        result.solve_time = solve_time;
        
    catch ME
        % Handle errors - return empty solutions
        x = [];
        y = [];
        info = struct();
        info.error_message = ME.message;
        result = create_sdpt3_result(info);
        result.solve_time = 0;  % Error case: no solve time
        
        fprintf('SDPT3 Runner: Error: %s\n', ME.message);
    end

end

function merged = merge_options(defaults, user_options)
    % Merge user options with defaults (user options take precedence)

    merged = defaults;
    if ~isempty(user_options) && isstruct(user_options)
        fields = fieldnames(user_options);
        for i = 1:length(fields)
            merged.(fields{i}) = user_options.(fields{i});
        end
    end

end

function version = get_sdpt3_version()
    % Get exact SDPT3 version dynamically from git submodule

    try
        if exist('sqlp', 'file') ~= 2
            version = 'SDPT3-Unknown';
            return;
        end
        
        % Get SDPT3 directory path
        sdpt3_path = which('sqlp');
        if ~isempty(sdpt3_path)
            [sdpt3_dir, ~, ~] = fileparts(sdpt3_path);
            
            % Try to get git tag/commit info from submodule
            try
                % Save current directory
                current_dir = pwd;
                
                % Change to SDPT3 directory and get git info
                cd(sdpt3_dir);
                [status, git_info] = system('git describe --tags --always 2>/dev/null');
                
                % Restore directory
                cd(current_dir);
                
                if status == 0 && ~isempty(strtrim(git_info))
                    git_info = strtrim(git_info);
                    version = sprintf('SDPT3-%s', git_info);
                    return;
                end
            catch ME
                % Git command failed, continue to README fallback
                fprintf('SDPT3 git version detection failed: %s\n', ME.message);
            end
            
            % Fallback: Try to read version from README
            readme_file = fullfile(sdpt3_dir, 'README');
            if exist(readme_file, 'file')
                fid = fopen(readme_file, 'r');
                if fid ~= -1
                    line = fgetl(fid);
                    fclose(fid);
                    if ischar(line) && contains(line, 'SDPT3 4.0')
                        version = 'SDPT3-4.0';
                        return;
                    end
                end
            end
        end
        
        version = 'SDPT3-Unknown';
    catch ME
        fprintf('SDPT3 version detection failed: %s\n', ME.message);
        version = 'SDPT3-Unknown';
    end

end

function result = create_sdpt3_result(info)
    % Create result structure from SDPT3 info
    result = struct();
    result.solver_name = 'SDPT3';
    result.solver_version = get_sdpt3_version();

    % Map SDPT3 status codes to standard format
    if isfield(info, 'error_message')
        result.status = 'error';
        result.termination_reason = 'Solver error';
        result.error_message = info.error_message;
    elseif isfield(info, 'termcode')
        switch info.termcode
            case 0
                result.status = 'optimal';
                result.termination_reason = 'Optimal solution found';
            case 1
                result.status = 'infeasible';
                result.termination_reason = 'Primal infeasible';
            case 2
                result.status = 'unbounded';
                result.termination_reason = 'Dual infeasible (primal unbounded)';
            case -1
                result.status = 'max_iter';
                result.termination_reason = 'Maximum iterations reached';
            case -2
                result.status = 'num_error';
                result.termination_reason = 'Numerical difficulties';
            case -3
                result.status = 'num_error';
                result.termination_reason = 'No progress in iterations';
            otherwise
                result.status = 'unknown';
                result.termination_reason = sprintf('Unknown termination code: %d', info.termcode);
        end
    else
        result.status = 'unknown';
        result.termination_reason = 'No termination code available';
    end

    % Extract iteration count
    if isfield(info, 'iter')
        result.iterations = info.iter;
    else
        result.iterations = NaN;
    end

    % Initialize other fields
    result.solve_time = NaN;
    result.setup_time = NaN;
    result.primal_objective_value = NaN;
    result.dual_objective_value = NaN;
    result.duality_gap = NaN;
    result.primal_infeasibility = NaN;
    result.dual_infeasibility = NaN;
    result.error_message = '';
end

