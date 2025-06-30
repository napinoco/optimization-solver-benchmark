function [A, b, c, K] = dat_loader(file_path)
% Load SDPLIB format .dat-s file and convert to SeDuMi format
%
% This function loads SDPLIB .dat-s files in SDPA sparse format and 
% converts them to standard SeDuMi format for use with MATLAB optimization solvers.
% 
% The implementation follows the same processing steps as the Python dat_loader.py:
% 1. Parse header (m, nblocks, block_sizes, c_vector)
% 2. Parse matrix entries into cell array structure
% 3. Convert to SeDuMi format problem data
%
% Input:
%   file_path: Path to .dat-s file containing SDPA sparse format data
%
% Output:
%   A: Constraint matrix (sparse, m x n)
%   b: Right-hand side vector (m x 1)
%   c: Objective vector (n x 1)
%   K: Cone structure (struct with fields K.f, K.l, K.q, K.s)
%
% Examples:
%   [A, b, c, K] = dat_loader('problems/SDPLIB/data/arch0.dat-s');
%   [A, b, c, K] = dat_loader('problems/SDPLIB/data/control1.dat-s');

try
    % Validate input
    if nargin < 1
        error('dat_loader:InvalidInput', 'File path is required');
    end
    
    if ~ischar(file_path) && ~isstring(file_path)
        error('dat_loader:InvalidInput', 'File path must be a string or char array');
    end
    
    % Convert to char if string
    file_path = char(file_path);
    
    % Check if file exists
    if ~exist(file_path, 'file')
        error('dat_loader:FileNotFound', 'File not found: %s', file_path);
    end
    
    fprintf('dat_loader: Parsing SDPA file manually...\n');
    
    % Step 1: Parse SDPA file (same as Python parse_sdpa_file)
    parsed_data = parse_sdpa_file(file_path);
    
    % Step 2: Convert to problem data (same as Python convert_to_problem_data)
    [A, b, c, K] = convert_to_problem_data(parsed_data);
    
    % Final validation and reporting
    fprintf('dat_loader: Successfully loaded SDPLIB problem\n');
    fprintf('  Variables (n): %d\n', size(A, 2));
    fprintf('  Constraints (m): %d\n', size(A, 1));
    fprintf('  Cone structure:\n');
    if isfield(K, 'f') && K.f > 0
        fprintf('    Free variables: %d\n', K.f);
    end
    if isfield(K, 'l') && K.l > 0
        fprintf('    Linear variables: %d\n', K.l);
    end
    if isfield(K, 'q') && ~isempty(K.q)
        fprintf('    SOC cones: %d\n', length(K.q));
    end
    if isfield(K, 's') && ~isempty(K.s)
        fprintf('    SDP blocks: %d (sizes: %s)\n', length(K.s), mat2str(K.s));
    end
    fprintf('  Constraint matrix density: %.2f%%\n', 100 * nnz(A) / numel(A));
    
catch ME
    % Provide informative error messages
    fprintf('dat_loader: Error loading file %s\n', file_path);
    fprintf('Error: %s\n', ME.message);
    
    % Re-throw with additional context
    error('dat_loader:LoadFailed', 'Failed to load .dat-s file: %s\nOriginal error: %s', ...
        file_path, ME.message);
end

end

function parsed_data = parse_sdpa_file(file_path)
% Parse SDPA sparse format file (equivalent to Python parse_sdpa_file)
%
% Args:
%   file_path: Path to the .dat-s file
%
% Returns:
%   struct containing parsed SDPA problem data:
%     - m: Number of constraints
%     - nblocks: Number of blocks  
%     - block_sizes: Vector of block sizes
%     - c: Objective vector
%     - matrices: Cell array of matrix entries

% Open file for reading
fid = fopen(file_path, 'r');
if fid == -1
    error('dat_loader:FileOpenError', 'Cannot open file: %s', file_path);
end

try
    % Read all lines and filter out comments
    data_lines = {};
    line_count = 0;
    
    % Read header lines (first 4 non-comment lines)
    while ~feof(fid) && line_count < 4
        line = fgetl(fid);
        if ischar(line)
            line = strtrim(line);
            % Skip empty lines and comment lines (starting with " or *)
            if ~isempty(line) && line(1) ~= '"' && line(1) ~= '*'
                line_count = line_count + 1;
                data_lines{line_count} = line;
            end
        end
    end
    
    if line_count < 4
        error('dat_loader:InvalidFormat', 'Invalid SDPA format: insufficient data lines');
    end
    
    % Parse header information
    % Line 1: m (number of constraints) 
    m = str2double(strtrim(data_lines{1}));
    if isnan(m) || m < 0
        error('dat_loader:InvalidFormat', 'Invalid number of constraints: %s', data_lines{1});
    end
    
    % Line 2: nblocks (number of blocks)
    nblocks = str2double(strtrim(data_lines{2}));
    if isnan(nblocks) || nblocks < 1
        error('dat_loader:InvalidFormat', 'Invalid number of blocks: %s', data_lines{2});
    end
    
    % Line 3: block sizes (remove punctuation like Python)
    block_sizes_line = data_lines{3};
    % Remove punctuation characters (same as Python)
    punctuation = ',(){}';
    for i = 1:length(punctuation)
        block_sizes_line = strrep(block_sizes_line, punctuation(i), ' ');
    end
    block_sizes_str = strsplit(strtrim(block_sizes_line));
    
    % Convert to numbers and validate
    block_sizes = zeros(1, nblocks);
    valid_count = 0;
    for i = 1:length(block_sizes_str)
        if ~isempty(strtrim(block_sizes_str{i}))
            valid_count = valid_count + 1;
            if valid_count <= nblocks
                block_sizes(valid_count) = str2double(block_sizes_str{i});
                if isnan(block_sizes(valid_count))
                    error('dat_loader:InvalidFormat', 'Invalid block size: %s', block_sizes_str{i});
                end
            end
        end
    end
    
    if valid_count ~= nblocks
        error('dat_loader:InvalidFormat', 'Block sizes count (%d) doesn''t match nblocks (%d)', valid_count, nblocks);
    end
    
    % Line 4: objective vector c
    c_str = strsplit(data_lines{4});
    c = zeros(m, 1);
    for i = 1:m
        if i <= length(c_str)
            c(i) = str2double(c_str{i});
            if isnan(c(i))
                error('dat_loader:InvalidFormat', 'Invalid objective coefficient: %s', c_str{i});
            end
        else
            error('dat_loader:InvalidFormat', 'Objective vector length (%d) doesn''t match m (%d)', length(c_str), m);
        end
    end
    
    % Initialize matrices cell array (same structure as Python)
    % matrices{matno+1}{blkno} contains entries for matrix matno, block blkno
    matrices = cell(m + 1, 1);
    for matno = 1:(m + 1)
        matrices{matno} = cell(nblocks, 1);
        for blkno = 1:nblocks
            matrices{matno}{blkno} = struct('i', [], 'j', [], 'val', []);
        end
    end
    
    % Parse matrix entries (same logic as Python)
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            line = strtrim(line);
            if ~isempty(line)
                % Parse: matno blkno i j value
                data = sscanf(line, '%d %d %d %d %f');
                if length(data) >= 5
                    matno = data(1);
                    blkno = data(2);
                    i = data(3);
                    j = data(4);
                    value = data(5);
                    
                    % Validate indices
                    if matno >= 0 && matno <= m && blkno >= 1 && blkno <= nblocks
                        % Store entry (keep 1-based indexing for MATLAB)
                        matrices{matno + 1}{blkno}.i(end + 1) = i;
                        matrices{matno + 1}{blkno}.j(end + 1) = j;
                        matrices{matno + 1}{blkno}.val(end + 1) = value;
                        
                        % Add symmetric entry if i != j
                        if i ~= j
                            matrices{matno + 1}{blkno}.i(end + 1) = j;
                            matrices{matno + 1}{blkno}.j(end + 1) = i;
                            matrices{matno + 1}{blkno}.val(end + 1) = value;
                        end
                    end
                end
            end
        end
    end
    
    fclose(fid);
    
    % Return parsed data structure
    parsed_data = struct();
    parsed_data.m = m;
    parsed_data.nblocks = nblocks;
    parsed_data.block_sizes = block_sizes;
    parsed_data.c = c;
    parsed_data.matrices = matrices;
    
catch ME
    if fid ~= -1
        fclose(fid);
    end
    rethrow(ME);
end

end

function [A, b, c, K] = convert_to_problem_data(parsed_data)
% Convert parsed SDPA data to SeDuMi format (equivalent to Python convert_to_problem_data)
%
% Args:
%   parsed_data: Parsed SDPA problem data from parse_sdpa_file
%
% Returns:
%   A: Constraint matrix (sparse, m x n)
%   b: Right-hand side vector (m x 1) 
%   c: Objective vector (n x 1)
%   K: Cone structure for SeDuMi

% Extract data
m = parsed_data.m;
nblocks = parsed_data.nblocks;
block_sizes = parsed_data.block_sizes;
c_sdpa = parsed_data.c;
matrices = parsed_data.matrices;

% Convert matrices from sparse entries to sparse matrices (same as Python logic)
for matno = 1:(m + 1)
    for blkno = 1:nblocks
        entries = matrices{matno}{blkno};
        if ~isempty(entries.val)
            % Create sparse matrix from entries (already 1-based indexing)
            block_size = abs(block_sizes(blkno));
            mat = sparse(entries.i, entries.j, entries.val, block_size, block_size);
            matrices{matno}{blkno} = mat;
        else
            % Empty matrix
            block_size = abs(block_sizes(blkno));
            matrices{matno}{blkno} = sparse(block_size, block_size);
        end
    end
end

% Build objective vector c (from F0, matno=0)
% Following Python logic: c = -hstack([matrices[0][blkno].reshape(1, -1) for blkno in range(nblocks)])
c_parts = cell(nblocks, 1);
for blkno = 1:nblocks
    mat = matrices{1}{blkno};  % matno=0 -> index 1
    if block_sizes(blkno) > 0
        % SDP block: reshape to row vector
        c_parts{blkno} = -mat(:)';  % Transpose to row vector and negate
    else
        % Diagonal block: extract diagonal
        c_parts{blkno} = -diag(mat)';  % Transpose to row vector and negate
    end
end
c = [c_parts{:}]';  % Concatenate and transpose to column vector

% Build constraint matrix A (from F1, F2, ..., Fm)
% Following Python logic: A = -vstack([hstack([matrices[matno][blkno].reshape(1, -1) for blkno in range(nblocks)]) for matno in range(1, m+1)])
A_rows = cell(m, 1);
for matno = 2:(m + 1)  % matno=1,2,...,m -> indices 2,3,...,m+1
    A_parts = cell(nblocks, 1);
    for blkno = 1:nblocks
        mat = matrices{matno}{blkno};
        if block_sizes(blkno) > 0
            % SDP block: reshape to row vector
            A_parts{blkno} = -mat(:)';  % Transpose to row vector and negate
        else
            % Diagonal block: extract diagonal  
            A_parts{blkno} = -diag(mat)';  % Transpose to row vector and negate
        end
    end
    A_rows{matno - 1} = [A_parts{:}];  % Concatenate horizontally
end

% Stack rows vertically to form A matrix
if ~isempty(A_rows)
    A = sparse(vertcat(A_rows{:}));
else
    A = sparse(m, length(c));
end

% Build right-hand side vector b (same as Python: b = -c_sdpa)
b = -c_sdpa;

% Analyze cone structure (same as Python logic)
% Convert block_sizes to individual SDP block sizes
sdp_blocks = [];
for block_size = block_sizes
    if block_size > 0
        sdp_blocks(end + 1) = block_size;
    else
        % Diagonal block becomes individual linear variables
        % But since we're dealing with SDP problems, we'll treat as 1x1 SDP blocks
        for i = 1:abs(block_size)
            sdp_blocks(end + 1) = 1;
        end
    end
end

% Create cone structure K for SeDuMi format
K = struct();
K.f = 0;  % No free variables
K.l = 0;  % No linear inequality constraints
K.q = []; % No second-order cone constraints
K.s = sdp_blocks;  % SDP block sizes

end