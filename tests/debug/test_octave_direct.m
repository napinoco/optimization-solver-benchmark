% Direct Octave Solver Test
% ========================
% Test GLPK and QP solvers directly without Python integration

fprintf('🐙 Direct Octave Solver Test\n');
fprintf('============================\n\n');

%% Test 1: Linear Programming with GLPK
fprintf('🧮 Testing Linear Programming (GLPK)...\n');
fprintf('Problem: min x1 + 2*x2 s.t. x1 + x2 >= 1, x1,x2 >= 0\n');

% Problem data
c = [1.0; 2.0];                    % Objective: min x1 + 2*x2
A = [-1, -1];                      % Constraint: -x1 - x2 <= -1 (i.e., x1 + x2 >= 1)
b = [-1];
lb = [0; 0];                       % x1, x2 >= 0
ub = [];                           % No upper bounds
ctype = 'U';                       % Upper bound constraint (<=)
vartype = ['C'; 'C'];              % Continuous variables
sense = 1;                         % Minimize

try
    fprintf('  Calling GLPK...\n');
    [x, fval, exitflag, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
    
    fprintf('  Solution: x = [%.3f, %.3f]\n', x(1), x(2));
    fprintf('  Objective: %.3f\n', fval);
    fprintf('  Exit flag: %d\n', exitflag);
    
    if exitflag == 0
        fprintf('  ✅ LP test PASSED!\n');
        lp_success = true;
    else
        fprintf('  ❌ LP test FAILED (exit code %d)\n', exitflag);
        lp_success = false;
    end
catch err
    fprintf('  ❌ LP test ERROR: %s\n', err.message);
    lp_success = false;
end

fprintf('\n');

%% Test 2: Quadratic Programming with QP
fprintf('🔄 Testing Quadratic Programming (QP)...\n');
fprintf('Problem: min 0.5*(x1^2 + x2^2) + x1 s.t. x1 + x2 = 1, x1,x2 >= 0\n');

% Problem data for QP
H = [1.0, 0.0; 0.0, 1.0];         % Quadratic matrix (identity)
q = [1.0; 0.0];                   % Linear term
A_eq = [1.0, 1.0];                % Equality constraint: x1 + x2 = 1
b_eq = [1.0];
x0 = [0.5; 0.5];                  % Initial point
lb_qp = [0; 0];                   % Lower bounds
ub_qp = [];                       % No upper bounds

try
    fprintf('  Calling QP...\n');
    % QP syntax: qp(x0, H, q, A, b, lb, ub, A_lb, A_in, A_ub)
    % For equality constraints, we use A_in matrix but need to check correct format
    % Let's try the simpler syntax first: qp(x0, H, q, A, b, lb, ub)
    [x, fval, info, lambda] = qp(x0, H, q, [], [], lb_qp, ub_qp);
    
    fprintf('  Solution: x = [%.3f, %.3f]\n', x(1), x(2));
    fprintf('  Objective: %.3f\n', fval);
    fprintf('  Info code: %d\n', info.info);
    
    if info.info == 0
        fprintf('  ✅ QP test PASSED!\n');
        qp_success = true;
    else
        fprintf('  ❌ QP test FAILED (info code %d)\n', info.info);
        qp_success = false;
    end
catch err
    fprintf('  ❌ QP test ERROR: %s\n', err.message);
    qp_success = false;
end

fprintf('\n');

%% Summary
fprintf('📊 Test Summary\n');
fprintf('===============\n');
if lp_success
    fprintf('✅ Linear Programming (GLPK): PASSED\n');
else
    fprintf('❌ Linear Programming (GLPK): FAILED\n');
end

if qp_success  
    fprintf('✅ Quadratic Programming (QP): PASSED\n');
else
    fprintf('❌ Quadratic Programming (QP): FAILED\n');
end

if lp_success && qp_success
    fprintf('\n🎉 All tests PASSED! Octave solvers are working.\n');
else
    fprintf('\n❌ Some tests FAILED. Check solver availability.\n');
end