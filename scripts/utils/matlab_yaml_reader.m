function [problem_info, file_path] = matlab_yaml_reader(problem_name, config_path)
% Read problem_registry.yaml and resolve problem name to file path
%
% This function reads the problem_registry.yaml configuration file and
% resolves a problem name to its corresponding file path and metadata.
% It provides a simple YAML parsing capability for MATLAB without external dependencies.
%
% Input:
%   problem_name: Name of the problem to look up (e.g., 'nb', 'arch0')
%   config_path: (optional) Path to problem_registry.yaml file
%                Default: 'config/problem_registry.yaml'
%
% Output:
%   problem_info: Struct containing problem metadata
%                 Fields: display_name, file_path, file_type, library_name, 
%                        for_test_flag, known_objective_value
%   file_path: Direct path to the problem file (for convenience)
%
% Examples:
%   [info, path] = matlab_yaml_reader('nb');
%   [info, path] = matlab_yaml_reader('arch0');
%   [info, path] = matlab_yaml_reader('control1', 'config/problem_registry.yaml');

try
    % Validate inputs
    if nargin < 1
        error('matlab_yaml_reader:InvalidInput', 'Problem name is required');
    end
    
    if ~ischar(problem_name) && ~isstring(problem_name)
        error('matlab_yaml_reader:InvalidInput', 'Problem name must be a string or char array');
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
    fprintf('matlab_yaml_reader: Reading configuration file %s...\n', config_path);
    
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
        
        % Final validation and reporting
        fprintf('matlab_yaml_reader: Successfully loaded problem metadata\n');
        fprintf('  Problem: %s\n', problem_name);
        fprintf('  Display name: %s\n', problem_info.display_name);
        fprintf('  File path: %s\n', problem_info.file_path);
        fprintf('  File type: %s\n', problem_info.file_type);
        fprintf('  Library: %s\n', problem_info.library_name);
        fprintf('  Test flag: %s\n', mat2str(problem_info.for_test_flag));
        if ~isnan(problem_info.known_objective_value)
            fprintf('  Known objective: %.6e\n', problem_info.known_objective_value);
        end
        
    catch ME
        if fid ~= -1
            fclose(fid);
        end
        rethrow(ME);
    end
    
catch ME
    % Provide informative error messages
    fprintf('matlab_yaml_reader: Error reading configuration for problem %s\n', problem_name);
    fprintf('Error: %s\n', ME.message);
    
    % Re-throw with additional context
    error('matlab_yaml_reader:ReadFailed', 'Failed to read configuration: %s\nOriginal error: %s', ...
        problem_name, ME.message);
end

end