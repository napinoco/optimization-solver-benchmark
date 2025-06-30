function success = save_json_result(result, output_file)
% Save MATLAB solver result to JSON file compatible with Python parsing
%
% This function converts a MATLAB solver result structure to JSON format
% and saves it to a file. The JSON format is compatible with Python's
% json.load() function and follows the standardized result schema.
%
% Input:
%   result: Solver result struct with standardized fields
%   output_file: Path to output JSON file
%
% Output:
%   success: Boolean indicating whether save operation succeeded

success = false;

try
    % Validate inputs
    if ~isstruct(result)
        error('save_json_result:InvalidInput', 'Result must be a struct');
    end
    
    if ~(ischar(output_file) || isstring(output_file))
        error('save_json_result:InvalidInput', 'Output file must be a string');
    end
    
    fprintf('save_json_result: Converting result to JSON format...\n');
    
    % Convert result to JSON string
    json_str = matlab_json_formatter(result);
    
    fprintf('save_json_result: Saving to file: %s\n', output_file);
    
    % Ensure output directory exists
    [output_dir, ~, ~] = fileparts(output_file);
    if ~isempty(output_dir) && ~exist(output_dir, 'dir')
        mkdir(output_dir);
        fprintf('save_json_result: Created directory: %s\n', output_dir);
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
                fprintf('save_json_result: Successfully saved JSON result (%d bytes)\n', file_info.bytes);
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
    fprintf('save_json_result: Error saving JSON result: %s\n', ME.message);
    
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
        
        error_json = matlab_json_formatter(error_result);
        
        fid = fopen(output_file, 'w');
        if fid ~= -1
            fprintf(fid, '%s', error_json);
            fclose(fid);
            fprintf('save_json_result: Saved error result to file\n');
        end
        
    catch
        % Final fallback - even error saving failed
        fprintf('save_json_result: Failed to save error information\n');
    end
end

end

function field_value = get_field_safe(struct_input, field_name, default_value)
% Safely get field value from struct with default fallback

if isstruct(struct_input) && isfield(struct_input, field_name)
    field_value = struct_input.(field_name);
else
    field_value = default_value;
end

end