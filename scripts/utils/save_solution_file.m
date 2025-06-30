function save_solution_file(problem_name, solver_name, x, y, save_solutions)
% Save solution vectors to .mat file if save_solutions is true
%
% Args:
%   problem_name: Name of the problem
%   solver_name: Name of the solver  
%   x: Primal solution vector
%   y: Dual solution vector
%   save_solutions: Boolean flag to control saving
%
% Output:
%   Creates {problem_name}_{solver_name}.mat in problems/solutions/ directory

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