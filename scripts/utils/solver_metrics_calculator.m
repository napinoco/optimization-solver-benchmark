function result = solver_metrics_calculator(result, x, y, A, b, c, K)
% Calculate objective values and infeasibility measures
% This is a shared function used by both SeDuMi and SDPT3 runners
%
% Args:
%   result: Existing result structure to update
%   x: Primal solution vector
%   y: Dual solution vector  
%   A: Constraint matrix
%   b: Right-hand side vector
%   c: Objective vector
%   K: Cone structure (SeDuMi format)
%
% Returns:
%   result: Updated result structure with calculated metrics

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

% Calculate infeasibility measures (same logic as Python)
if ~isempty(x) && ~isempty(y) && ~isempty(A) && ~isempty(b) && ~isempty(c)
    try
        % Primal infeasibility: ||A*x - b|| / (1 + ||b||)
        x_vec = x(:);
        b_vec = b(:);
        if size(A, 2) == length(x_vec) && size(A, 1) == length(b_vec)
            primal_residual = A * x_vec - b_vec;
            result.primal_infeasibility = norm(primal_residual) / (1 + norm(b_vec));
        end
        
        % Dual infeasibility: cone-specific calculation (matching Python logic)
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
% dinf2 = 0  # similar to np.sum(constraint.violation() ** 2 for constraint in cvx_problem.constraints)

dinf2 = 0;
nvar_cnt = 0;

% if 'free_vars' in cone_structure:
%     free_vars = cone_structure['free_vars']
%     if free_vars:
%         begin = nvar_cnt
%         end = nvar_cnt + free_vars
%         dinf2 += np.linalg.norm(cmAty[begin:end], ord=2) ** 2
%         nvar_cnt = end
if isfield(K, 'f') && K.f > 0
    free_vars = K.f;
    begin = nvar_cnt + 1;  % MATLAB 1-based indexing
    ending = nvar_cnt + free_vars;
    dinf2 = dinf2 + norm(cmAty(begin:ending), 2)^2;
    nvar_cnt = ending;
end

% if 'nonneg_vars' in cone_structure:
%     nonneg_vars = cone_structure['nonneg_vars']
%     if nonneg_vars:
%         begin = nvar_cnt
%         end = nvar_cnt + nonneg_vars
%         dinf2 += np.linalg.norm(np.minimum(cmAty[begin:end], 0), ord=2) ** 2
%         nvar_cnt = end
if isfield(K, 'l') && K.l > 0
    nonneg_vars = K.l;
    begin = nvar_cnt + 1;  % MATLAB 1-based indexing
    ending = nvar_cnt + nonneg_vars;
    dinf2 = dinf2 + norm(min(cmAty(begin:ending), 0), 2)^2;
    nvar_cnt = ending;
end

% if 'soc_cones' in cone_structure:
%     soc_cones = cone_structure['soc_cones']
%     for ndim in soc_cones:
%         if ndim <= 0:
%             continue
%         begin = nvar_cnt
%         end = nvar_cnt + ndim
%         dinf2 += np.linalg.norm(proj_onto_soc(-cmAty[begin:end]), ord=2) ** 2  # Pi_{K*}(-z) = z - Pi_K(z)
%         nvar_cnt = end
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

% if 'sdp_cones' in cone_structure:
%     sdp_cones = cone_structure['sdp_cones']
%     for ndim in sdp_cones:
%         if ndim <= 0:
%             continue
%         begin = nvar_cnt
%         end = nvar_cnt + ndim * ndim
%         eigvals = np.linalg.eigvalsh(cmAty[begin:end].reshape(ndim, ndim))
%         neg_eigvals = np.minimum(eigvals, 0)
%         dinf2 += np.sum(neg_eigvals ** 2)
%         nvar_cnt = end
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
% Project onto second-order cone (matching Python logic)
% def proj_onto_soc(z):
%     z0 = z[0]
%     znorm = np.linalg.norm(z[1:], ord=2)
%     if znorm <= z0:
%         return z
%     elif znorm <= -z0:
%         return np.zeros_like(z)
%     else:
%         scale = (z0 + znorm) / 2
%         return np.concatenate(([1], z[1:] / znorm)) * scale

z0 = z(1);  % MATLAB 1-based indexing: z[0] becomes z(1)
znorm = norm(z(2:end), 2);  % z[1:] becomes z(2:end)
if znorm <= z0
    proj_z = z;
elseif znorm <= -z0
    proj_z = zeros(size(z));
else
    scale = (z0 + znorm) / 2;
    proj_z = [1; z(2:end) / znorm] * scale;
end

end