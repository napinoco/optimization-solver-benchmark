function [A, b, c, K] = mps_loader(file_path)
    % Load NETLIB format .mps file and convert to SeDuMi format
    %
    % This function loads MPS (Mathematical Programming System) files and
    % converts them to standard SeDuMi format for use with MATLAB optimization
    % solvers. It mirrors the conversion performed by the Python loader
    % (scripts/data_loaders/python/mps_loader.py): inequality constraints are
    % converted to equality constraints via slack variables, and variable
    % bounds are handled through the cone structure.
    %
    % Unlike the Python loader, variables are reordered so that free variables
    % (K.f) come strictly before nonnegative variables (K.l), as required by
    % SeDuMi's cone convention.
    %
    % Input:
    %   file_path: Path to .mps file
    %
    % Output:
    %   A: Constraint matrix (sparse, m x n)
    %   b: Right-hand side vector (m x 1)
    %   c: Objective vector (n x 1)
    %   K: Cone structure (struct with fields K.f, K.l, K.q, K.s)
    %
    % Example:
    %   [A, b, c, K] = mps_loader('problems/NETLIB/mps_files/afiro.mps');

    try
        if nargin < 1
            error('mps_loader:InvalidInput', 'File path is required');
        end

        if ~ischar(file_path) && ~isstring(file_path)
            error('mps_loader:InvalidInput', 'File path must be a string or char array');
        end

        file_path = char(file_path);

        if ~exist(file_path, 'file')
            error('mps_loader:FileNotFound', 'File not found: %s', file_path);
        end

        fprintf('mps_loader: Parsing MPS file...\n');

        parsed = parse_mps_file(file_path);
        [A, b, c, K] = convert_to_sedumi(parsed);

        fprintf('mps_loader: Successfully loaded MPS problem\n');
        fprintf('  Variables (n): %d\n', size(A, 2));
        fprintf('  Constraints (m): %d\n', size(A, 1));
        fprintf('  Cone structure: K.f=%d, K.l=%d\n', K.f, K.l);

    catch ME
        fprintf('mps_loader: Error loading file %s\n', file_path);
        fprintf('Error: %s\n', ME.message);
        error('mps_loader:LoadFailed', 'Failed to load .mps file: %s\nOriginal error: %s', ...
            file_path, ME.message);
    end

end

function parsed = parse_mps_file(file_path)
    % Parse MPS format file into row/column/rhs/bounds structures.

    fid = fopen(file_path, 'r');
    if fid == -1
        error('mps_loader:FileOpenError', 'Cannot open file: %s', file_path);
    end

    row_order = {};
    row_type = containers.Map('KeyType', 'char', 'ValueType', 'char');
    row_index = containers.Map('KeyType', 'char', 'ValueType', 'double');
    obj_row_name = '';

    col_order = {};
    col_index = containers.Map('KeyType', 'char', 'ValueType', 'double');
    tri_i = [];
    tri_j = [];
    tri_v = [];

    rhs_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
    bound_lo = containers.Map('KeyType', 'char', 'ValueType', 'double');
    bound_up = containers.Map('KeyType', 'char', 'ValueType', 'double');

    current_section = '';

    try
        while ~feof(fid)
            line = fgetl(fid);
            if ~ischar(line)
                continue;
            end
            raw_line = line;
            trimmed = strtrim(line);
            if isempty(trimmed) || trimmed(1) == '*'
                continue;
            end

            upper_line = upper(trimmed);
            if startsWith(upper_line, 'NAME')
                current_section = 'NAME';
                continue;
            elseif strcmp(upper_line, 'ROWS')
                current_section = 'ROWS';
                continue;
            elseif strcmp(upper_line, 'COLUMNS')
                current_section = 'COLUMNS';
                continue;
            elseif strcmp(upper_line, 'RHS')
                current_section = 'RHS';
                continue;
            elseif strcmp(upper_line, 'BOUNDS')
                current_section = 'BOUNDS';
                continue;
            elseif strcmp(upper_line, 'RANGES')
                current_section = 'RANGES';
                continue;
            elseif strcmp(upper_line, 'ENDATA')
                break;
            end

            switch current_section
                case 'ROWS'
                    parts = strsplit(raw_line);
                    parts = parts(~cellfun(@isempty, parts));
                    if numel(parts) >= 2
                        r_type = upper(parts{1});
                        r_name = parts{2};
                        row_order{end + 1} = r_name; %#ok<AGROW>
                        row_type(r_name) = r_type;
                        row_index(r_name) = numel(row_order);
                        if strcmp(r_type, 'N') && isempty(obj_row_name)
                            obj_row_name = r_name;
                        end
                    end

                case 'COLUMNS'
                    parts = strsplit(raw_line);
                    parts = parts(~cellfun(@isempty, parts));
                    if numel(parts) < 3
                        continue;
                    end
                    c_name = parts{1};
                    if ~isKey(col_index, c_name)
                        col_order{end + 1} = c_name; %#ok<AGROW>
                        col_index(c_name) = numel(col_order);
                    end
                    j = col_index(c_name);
                    k = 2;
                    while k <= numel(parts) - 1
                        r_name = parts{k};
                        val = str2double(parts{k + 1});
                        if isnan(val) || ~isKey(row_index, r_name)
                            break;
                        end
                        tri_i(end + 1) = row_index(r_name); %#ok<AGROW>
                        tri_j(end + 1) = j; %#ok<AGROW>
                        tri_v(end + 1) = val; %#ok<AGROW>
                        k = k + 2;
                    end

                case 'RHS'
                    parts = strsplit(raw_line);
                    parts = parts(~cellfun(@isempty, parts));
                    if numel(parts) < 3
                        continue;
                    end
                    k = 2;
                    while k <= numel(parts) - 1
                        r_name = parts{k};
                        val = str2double(parts{k + 1});
                        if isnan(val)
                            break;
                        end
                        rhs_map(r_name) = val;
                        k = k + 2;
                    end

                case 'BOUNDS'
                    parts = strsplit(raw_line);
                    parts = parts(~cellfun(@isempty, parts));
                    if numel(parts) < 3
                        continue;
                    end
                    b_type = upper(parts{1});
                    c_name = parts{3};
                    if ~isKey(bound_lo, c_name)
                        bound_lo(c_name) = 0.0;
                        bound_up(c_name) = Inf;
                    end
                    switch b_type
                        case 'LO'
                            bound_lo(c_name) = str2double(parts{4});
                        case 'UP'
                            bound_up(c_name) = str2double(parts{4});
                        case 'FX'
                            v = str2double(parts{4});
                            bound_lo(c_name) = v;
                            bound_up(c_name) = v;
                        case 'FR'
                            bound_lo(c_name) = -Inf;
                            bound_up(c_name) = Inf;
                        case 'MI'
                            bound_lo(c_name) = -Inf;
                        case 'PL'
                            bound_up(c_name) = Inf;
                        case 'BV'
                            bound_lo(c_name) = 0.0;
                            bound_up(c_name) = 1.0;
                    end

                case 'RANGES'
                    % RANGES constraints are not applied (parity with the
                    % Python loader, which parses but never uses them).
                    continue;
            end
        end

        fclose(fid);
    catch ME
        if fid ~= -1
            fclose(fid);
        end
        rethrow(ME);
    end

    parsed = struct();
    parsed.row_order = {row_order{:}}; %#ok<CCAT1>
    parsed.row_type = row_type;
    parsed.row_index = row_index;
    parsed.obj_row_name = obj_row_name;
    parsed.col_order = {col_order{:}}; %#ok<CCAT1>
    parsed.col_index = col_index;
    parsed.tri_i = tri_i;
    parsed.tri_j = tri_j;
    parsed.tri_v = tri_v;
    parsed.rhs_map = rhs_map;
    parsed.bound_lo = bound_lo;
    parsed.bound_up = bound_up;

end

function [A, b, c, K] = convert_to_sedumi(parsed)
    % Convert parsed MPS data to SeDuMi standard form:
    %   min  c'x
    %   s.t. A x = b, x in K
    %
    % Inequality constraints are converted to equalities with slack
    % variables; variable bounds are handled through the cone structure.
    % Variables are ordered [free vars][nonnegative vars] as required by
    % SeDuMi (nonnegative vars include original bounded vars and all slacks).

    n_orig_cols = numel(parsed.col_order);
    n_rows = numel(parsed.row_order);

    Coef = sparse(parsed.tri_i, parsed.tri_j, parsed.tri_v, n_rows, n_orig_cols);

    if ~isempty(parsed.obj_row_name)
        c_orig = full(Coef(parsed.row_index(parsed.obj_row_name), :))';
    else
        c_orig = zeros(n_orig_cols, 1);
    end

    eq_rows = {};
    le_rows = {};
    ge_rows = {};
    for i = 1:numel(parsed.row_order)
        r_name = parsed.row_order{i};
        r_type = parsed.row_type(r_name);
        if strcmp(r_type, 'E')
            eq_rows{end + 1} = r_name; %#ok<AGROW>
        elseif strcmp(r_type, 'L')
            le_rows{end + 1} = r_name; %#ok<AGROW>
        elseif strcmp(r_type, 'G')
            ge_rows{end + 1} = r_name; %#ok<AGROW>
        end
    end

    lo = zeros(n_orig_cols, 1);
    up = Inf(n_orig_cols, 1);
    for j = 1:n_orig_cols
        c_name = parsed.col_order{j};
        if isKey(parsed.bound_lo, c_name)
            lo(j) = parsed.bound_lo(c_name);
        end
        if isKey(parsed.bound_up, c_name)
            up(j) = parsed.bound_up(c_name);
        end
    end

    n_eq = numel(eq_rows);
    n_le = numel(le_rows);
    n_ge = numel(ge_rows);
    ub_var_indices = find(up < Inf)';
    n_ub = numel(ub_var_indices);

    n_slack = n_le + n_ge + n_ub;
    n_total_vars = n_orig_cols + n_slack;

    row_idx_of = @(name) parsed.row_index(name);
    eq_idx = cellfun(row_idx_of, eq_rows);
    le_idx = cellfun(row_idx_of, le_rows);
    ge_idx = cellfun(row_idx_of, ge_rows);

    A_eq_orig = Coef(eq_idx, :);
    A_le_orig = Coef(le_idx, :);
    A_ge_orig = Coef(ge_idx, :);

    A_eq_full = [A_eq_orig, sparse(n_eq, n_slack)];
    A_le_full = [A_le_orig, speye(n_le), sparse(n_le, n_ge), sparse(n_le, n_ub)];
    A_ge_full = [A_ge_orig, sparse(n_ge, n_le), -speye(n_ge), sparse(n_ge, n_ub)];

    if n_ub > 0
        A_ub_orig = sparse(1:n_ub, ub_var_indices, 1, n_ub, n_orig_cols);
        A_ub_full = [A_ub_orig, sparse(n_ub, n_le), sparse(n_ub, n_ge), speye(n_ub)];
        b_ub = up(ub_var_indices)';
    else
        A_ub_full = sparse(0, n_total_vars);
        b_ub = zeros(0, 1);
    end

    A_full = [A_eq_full; A_le_full; A_ge_full; A_ub_full];

    b_eq = zeros(n_eq, 1);
    for i = 1:n_eq
        if isKey(parsed.rhs_map, eq_rows{i})
            b_eq(i) = parsed.rhs_map(eq_rows{i});
        end
    end
    b_le = zeros(n_le, 1);
    for i = 1:n_le
        if isKey(parsed.rhs_map, le_rows{i})
            b_le(i) = parsed.rhs_map(le_rows{i});
        end
    end
    b_ge = zeros(n_ge, 1);
    for i = 1:n_ge
        if isKey(parsed.rhs_map, ge_rows{i})
            b_ge(i) = parsed.rhs_map(ge_rows{i});
        end
    end

    b_full = [b_eq; b_le; b_ge; b_ub(:)];
    c_full = [c_orig; zeros(n_slack, 1)];

    % Shift bounded variables with a nonzero finite lower bound so that
    % x_new = x - lo >= 0 (free variables and default lo=0 need no shift).
    for j = 1:n_orig_cols
        if lo(j) ~= 0 && lo(j) > -Inf
            b_full = b_full - A_full(:, j) * lo(j);
        end
    end

    % Reorder columns: free variables first, then nonnegative (original
    % bounded vars + all slacks), matching SeDuMi's K.f/K.l convention.
    free_mask = false(1, n_total_vars);
    free_mask(1:n_orig_cols) = (lo == -Inf)';
    perm = [find(free_mask), find(~free_mask)];

    A = A_full(:, perm);
    c = c_full(perm);
    b = b_full;

    K = struct();
    K.f = sum(free_mask);
    K.l = n_total_vars - K.f;
    K.q = [];
    K.s = [];

end
