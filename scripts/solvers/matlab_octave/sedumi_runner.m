function [x, y, result] = sedumi_runner(A, b, c, K, options)
    % Execute SeDuMi solver and return solutions with result structure
    %
    % This function runs the SeDuMi optimization solver with minimal configuration
    % for fair benchmarking. It returns the solutions and a standardized result structure.
    %
    % Input:
    %   A: Constraint matrix (sparse, m x n)
    %   b: Right-hand side vector (m x 1)  
    %   c: Objective vector (n x 1)
    %   K: Cone structure (struct with fields K.f, K.l, K.q, K.s)
    %   options: (optional) Solver options struct
    %
    % Output:
    %   x: Primal solution vector
    %   y: Dual solution vector
    %   result: Standardized result structure

    % Initialize empty outputs
    x = [];
    y = [];
    result = struct();

    try
        % Validate inputs
        if nargin < 4
            error('sedumi_runner:InvalidInput', 'Insufficient input arguments. Need A, b, c, K');
        end
        
        % Validate problem dimensions
        [m, n] = size(A);
        if length(b) ~= m
            error('sedumi_runner:DimensionMismatch', 'Dimension mismatch: length(b) must equal size(A,1)');
        end
        if length(c) ~= n
            error('sedumi_runner:DimensionMismatch', 'Dimension mismatch: length(c) must equal size(A,2)');
        end
        
        % Set up minimal SeDuMi options for fair benchmarking
        if nargin < 5 || isempty(options)
            options = struct();
        end
        
        % Default SeDuMi options - minimal configuration for fair comparison
        default_options = struct();
        default_options.fid = 0;        % No output (silent mode)
        
        % Merge with user options (user options override defaults)
        pars = merge_options(default_options, options);
        
        % Ensure matrices are in correct format
        if ~issparse(A)
            A = sparse(A);
        end
        b = full(b(:));  % Column vector
        c = full(c(:));  % Column vector
        
        % Validate cone structure
        K = validate_cone_structure(K, n);
        
        % Call SeDuMi solver
        fprintf('SeDuMi: Starting solve with %d variables, %d constraints\n', n, m);
        
        % Measure solve time
        solve_start_time = tic;
        [x, y, info] = sedumi(A, b, c, K, pars);
        solve_time = toc(solve_start_time);
        
        fprintf('SeDuMi: Completed in %.3f seconds\n', solve_time);
        
        % Create result structure from SeDuMi info
        result = create_sedumi_result(info);
        result.solve_time = solve_time;
        
    catch ME
        % Handle errors - return empty solutions
        x = [];
        y = [];
        info = struct();
        info.error_message = ME.message;
        result = create_sedumi_result(info);
        result.solve_time = 0;  % Error case: no solve time
        
        fprintf('SeDuMi Runner: Error: %s\n', ME.message);
    end

end

function K = validate_cone_structure(K, n)
    % Validate and normalize cone structure for SeDuMi

    if ~isstruct(K)
        error('sedumi_runner:InvalidCone', 'Cone structure K must be a struct');
    end

    % Set default values
    if ~isfield(K, 'f')
        K.f = 0;
    end
    if ~isfield(K, 'l')
        K.l = 0;
    end
    if ~isfield(K, 'q')
        K.q = [];
    end
    if ~isfield(K, 's')
        K.s = [];
    end

    % Validate cone dimensions
    total_vars = K.f + K.l + sum(K.q) + sum(K.s .* K.s);
    if total_vars ~= n
        warning('sedumi_runner:ConeDimensionMismatch', ...
            'Cone structure dimensions (%d) do not match problem size (%d)', total_vars, n);
    end

    % Ensure correct format for SeDuMi
    K.f = max(0, K.f);
    K.l = max(0, K.l);
    if ~isempty(K.q)
        K.q = K.q(:)';  % Row vector
        K.q = K.q(K.q > 0);  % Remove zero/negative entries
    end
    if ~isempty(K.s)
        K.s = K.s(:)';  % Row vector  
        K.s = K.s(K.s > 0);  % Remove zero/negative entries
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

function version = get_sedumi_version()
    % Get exact SeDuMi version (no fallbacks)

    try
        if exist('sedumi_version', 'file') == 2
            v = sedumi_version();
            if ischar(v) && ~isempty(v)
                version = sprintf('SeDuMi-%s', v);
                return;
            end
        end
        version = 'SeDuMi-Unknown';
    catch
        version = 'SeDuMi-Unknown';
    end

end

function result = create_sedumi_result(info)
    % Create result structure from SeDuMi info
    result = struct();
    result.solver_name = 'SeDuMi';
    result.solver_version = get_sedumi_version();

    % Map SeDuMi status codes to standard format
    if isfield(info, 'error_message')
        result.status = 'error';
        result.termination_reason = 'Solver error';
        result.error_message = info.error_message;
    elseif isfield(info, 'pinf') && info.pinf == 1
        result.status = 'infeasible';
        result.termination_reason = 'Primal infeasible';
    elseif isfield(info, 'dinf') && info.dinf == 1
        result.status = 'unbounded';  
        result.termination_reason = 'Dual infeasible (primal unbounded)';
    elseif isfield(info, 'numerr') && info.numerr > 0
        result.status = 'num_error';
        result.termination_reason = sprintf('Numerical error (code: %d)', info.numerr);
    else
        % For SeDuMi, if pinf=0 and dinf=0 and numerr=0, solution is optimal
        if isfield(info, 'pinf') && info.pinf == 0 && isfield(info, 'dinf') && info.dinf == 0
            result.status = 'optimal';
            result.termination_reason = 'Optimal solution found';
        else
            result.status = 'unknown';
            result.termination_reason = 'Solution status unclear';
        end
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
    result.primal_objective = NaN;
    result.dual_objective = NaN;
    result.gap = NaN;
    result.primal_infeasibility = NaN;
    result.dual_infeasibility = NaN;
    result.error_message = '';
end

