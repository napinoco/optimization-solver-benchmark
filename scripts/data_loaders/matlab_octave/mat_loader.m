function [A, b, c, K] = mat_loader(file_path)
    % Load SeDuMi format .mat file and extract optimization problem data
    %
    % This function loads DIMACS .mat/.mat.gz files and converts them to
    % standard SeDuMi format for use with MATLAB optimization solvers.
    %
    % Input:
    %   file_path: Path to .mat or .mat.gz file containing SeDuMi format data
    %
    % Output:
    %   A: Constraint matrix (sparse, m x n)
    %   b: Right-hand side vector (m x 1)  
    %   c: Objective vector (n x 1)
    %   K: Cone structure (struct with fields like K.l, K.q, K.s)
    %
    % The function handles:
    % - Compressed .mat.gz files (automatic decompression)
    % - Standard .mat files
    % - DIMACS format conversion (At -> A transpose)
    % - Error handling for corrupted files
    % - Default cone structure construction when missing
    %
    % Examples:
    %   [A, b, c, K] = mat_loader('problems/DIMACS/data/ANTENNA/nb.mat.gz');
    %   [A, b, c, K] = mat_loader('problems/DIMACS/data/COPOS/copo14.mat.gz');
    
    try
        % Validate input
        if nargin < 1
            error('mat_loader:InvalidInput', 'File path is required');
        end
        
        if ~ischar(file_path) && ~isstring(file_path)
            error('mat_loader:InvalidInput', 'File path must be a string or char array');
        end
        
        % Convert to char if string
        file_path = char(file_path);
        
        % Check if file exists
        if ~exist(file_path, 'file')
            error('mat_loader:FileNotFound', 'File not found: %s', file_path);
        end
        
        % Load .mat file (MATLAB handles .gz automatically in recent versions)
        try
            data = load(file_path);
        catch load_error
            % If direct load fails, try manual decompression for .gz files
            if endsWith(file_path, '.gz')
                try
                    % Create temporary file for decompressed data
                    [~, temp_name] = fileparts(tempname);
                    temp_file = fullfile(tempdir, [temp_name, '.mat']);
                    
                    % Use system gunzip if available
                    if isunix || ismac
                        status = system(sprintf('gunzip -c "%s" > "%s"', file_path, temp_file));
                        if status ~= 0
                            error('Failed to decompress .gz file using gunzip');
                        end
                    else
                        % On Windows, try MATLAB's gzip functionality
                        error('Compressed file loading not supported on Windows. Please decompress manually.');
                    end
                    
                    % Load decompressed file
                    data = load(temp_file);
                    
                    % Clean up temporary file
                    if exist(temp_file, 'file')
                        delete(temp_file);
                    end
                    
                catch decomp_error
                    rethrow(load_error); % Rethrow original error if decompression fails
                end
            else
                rethrow(load_error);
            end
        end
        
        % Validate that we have the required SeDuMi format fields
        required_fields = {'c'};  % c is always required
        for i = 1:length(required_fields)
            field = required_fields{i};
            if ~isfield(data, field)
                error('mat_loader:InvalidFormat', 'Missing required field: %s', field);
            end
        end
        
        % Extract basic components
        c = data.c;
        
        % Validate c vector
        if ~isnumeric(c) || ~isvector(c)
            error('mat_loader:InvalidFormat', 'Objective vector c must be a numeric vector');
        end
        
        % Ensure c is a column vector
        if size(c, 2) > size(c, 1)
            c = c';
        end
        
        n = length(c);  % Number of variables
        
        % Extract constraint matrix A
        % DIMACS files may store At (transpose) instead of A
        if isfield(data, 'A')
            A = data.A;
            % Validate dimensions
            if size(A, 2) ~= n
                error('mat_loader:InvalidFormat', 'Constraint matrix A dimensions inconsistent with c');
            end
        elseif isfield(data, 'At')
            A = data.At';  % Transpose to get A from At
            % Validate dimensions
            if size(A, 2) ~= n
                error('mat_loader:InvalidFormat', 'Constraint matrix At dimensions inconsistent with c');
            end
        else
            % Create empty constraint matrix if none provided
            A = sparse(0, n);
            warning('mat_loader:NoConstraints', 'No constraint matrix found, using empty matrix');
        end
        
        m = size(A, 1);  % Number of constraints
        
        % Extract right-hand side vector b
        if isfield(data, 'b')
            b = data.b;
            % Validate b vector
            if ~isnumeric(b) || ~isvector(b)
                error('mat_loader:InvalidFormat', 'Right-hand side vector b must be a numeric vector');
            end
            
            % Ensure b is a column vector
            if size(b, 2) > size(b, 1)
                b = b';
            end
            
            % Validate dimensions
            if length(b) ~= m
                error('mat_loader:InvalidFormat', 'Right-hand side vector b dimensions inconsistent with A');
            end
        else
            % Create zero right-hand side if none provided
            b = zeros(m, 1);
            if m > 0
                warning('mat_loader:NoRHS', 'No right-hand side vector found, using zeros');
            end
        end
        
        % Extract or construct cone structure K
        if isfield(data, 'K')
            K = data.K;
            
            % Validate that K is a struct
            if ~isstruct(K)
                error('mat_loader:InvalidFormat', 'Cone structure K must be a struct');
            end
            
            % Validate cone structure consistency
            total_vars = 0;
            
            % Free variables
            if isfield(K, 'f')
                if ~isscalar(K.f) || K.f < 0 || K.f ~= round(K.f)
                    error('mat_loader:InvalidFormat', 'K.f (free variables) must be a non-negative integer');
                end
                total_vars = total_vars + K.f;
            else
                K.f = 0;
            end
            
            % Linear inequality variables (x >= 0)
            if isfield(K, 'l')
                % Handle empty array case (convert [] to 0)
                if isempty(K.l)
                    K.l = 0;
                elseif ~isscalar(K.l) || K.l < 0 || K.l ~= round(K.l)
                    error('mat_loader:InvalidFormat', 'K.l (linear variables) must be a non-negative integer');
                end
                total_vars = total_vars + K.l;
            else
                K.l = 0;
            end
            
            % Second-order cone variables
            if isfield(K, 'q')
                % Handle empty array case (keep as empty for no SOCP cones)
                if ~isempty(K.q) && (~isnumeric(K.q) || any(K.q < 0) || any(K.q ~= round(K.q)))
                    error('mat_loader:InvalidFormat', 'K.q (SOCP cone sizes) must be non-negative integers');
                end
                total_vars = total_vars + sum(K.q);
            end
            
            % Semidefinite cone variables
            if isfield(K, 's')
                if ~isnumeric(K.s) || any(K.s < 0) || any(K.s ~= round(K.s))
                    error('mat_loader:InvalidFormat', 'K.s (SDP block sizes) must be non-negative integers');
                end
                total_vars = total_vars + sum(K.s .* K.s);
            end
            
            % Validate total variable count
            if total_vars > 0 && total_vars ~= n
                warning('mat_loader:DimensionMismatch', ...
                    'Cone structure variables (%d) do not match objective vector length (%d)', ...
                    total_vars, n);
            end
            
        else
            % Construct default cone structure
            % Assume all variables are linear inequality (x >= 0)
            K = struct();
            K.f = 0;          % No free variables
            K.l = n;          % All variables are linear inequality
            
            warning('mat_loader:DefaultCone', ...
                'No cone structure found, assuming all %d variables are linear (x >= 0)', n);
        end
        
        % Convert to full sparse matrices if needed
        if ~issparse(A)
            A = sparse(A);
        end
        
        % Final validation
        fprintf('mat_loader: Successfully loaded problem\n');
        fprintf('  Variables (n): %d\n', n);
        fprintf('  Constraints (m): %d\n', m);
        fprintf('  Cone structure:\n');
        if isfield(K, 'f') && K.f > 0
            fprintf('    Free variables: %d\n', K.f);
        end
        if isfield(K, 'l') && K.l > 0
            fprintf('    Linear variables: %d\n', K.l);
        end
        if isfield(K, 'q') && ~isempty(K.q)
            fprintf('    SOCP cones: %d (sizes: %s)\n', length(K.q), mat2str(K.q));
        end
        if isfield(K, 's') && ~isempty(K.s)
            fprintf('    SDP blocks: %d (sizes: %s)\n', length(K.s), mat2str(K.s));
        end
        
    catch ME
        % Provide informative error messages
        fprintf('mat_loader: Error loading file %s\n', file_path);
        fprintf('Error: %s\n', ME.message);
        
        % Re-throw with additional context
        error('mat_loader:LoadFailed', 'Failed to load .mat file: %s\nOriginal error: %s', ...
            file_path, ME.message);
    end
    
end