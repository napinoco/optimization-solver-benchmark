function data = loadjson(filename)
% LOADJSON Load JSON data from file
% 
% Usage:
%   data = loadjson(filename)
%
% This function loads JSON data from a file and converts it to Octave/MATLAB
% structures. Provides basic JSON parsing functionality.

    % Check if file exists
    if ~exist(filename, 'file')
        error('File does not exist: %s', filename);
    end
    
    % Read file content
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    json_str = fread(fid, '*char')';
    fclose(fid);
    
    % Parse JSON string
    try
        if exist('jsondecode', 'file')
            % Use built-in JSON decoder if available (Octave 4.4+)
            data = jsondecode(json_str);
        else
            % Fallback to basic parsing
            data = parse_json_basic(json_str);
        end
    catch err
        error('Failed to parse JSON: %s', err.message);
    end
end

function data = parse_json_basic(json_str)
% Basic JSON parser for simple structures
% This is a simplified parser for basic JSON objects and arrays

    % Remove whitespace
    json_str = strtrim(json_str);
    
    if isempty(json_str)
        data = [];
        return;
    end
    
    % Parse based on first character
    switch json_str(1)
        case '{'
            data = parse_object(json_str);
        case '['
            data = parse_array(json_str);
        case '"'
            data = parse_string(json_str);
        otherwise
            data = parse_value(json_str);
    end
end

function obj = parse_object(json_str)
% Parse JSON object
    obj = struct();
    
    if length(json_str) < 2 || json_str(1) ~= '{' || json_str(end) ~= '}'
        error('Invalid JSON object format');
    end
    
    % Remove braces
    content = json_str(2:end-1);
    content = strtrim(content);
    
    if isempty(content)
        return;
    end
    
    % Split by commas (simple approach)
    pairs = split_by_comma(content);
    
    for i = 1:length(pairs)
        pair = strtrim(pairs{i});
        colon_pos = find(pair == ':', 1);
        
        if isempty(colon_pos)
            continue;
        end
        
        key_str = strtrim(pair(1:colon_pos-1));
        value_str = strtrim(pair(colon_pos+1:end));
        
        % Parse key (remove quotes)
        if key_str(1) == '"' && key_str(end) == '"'
            key = key_str(2:end-1);
        else
            key = key_str;
        end
        
        % Parse value
        value = parse_json_basic(value_str);
        
        % Store in struct (convert invalid field names for Octave compatibility)
        key = makeValidName(key);
        obj.(key) = value;
    end
end

function arr = parse_array(json_str)
% Parse JSON array
    if length(json_str) < 2 || json_str(1) ~= '[' || json_str(end) ~= ']'
        error('Invalid JSON array format');
    end
    
    % Remove brackets
    content = json_str(2:end-1);
    content = strtrim(content);
    
    if isempty(content)
        arr = {};
        return;
    end
    
    % Split by commas
    elements = split_by_comma(content);
    arr = cell(1, length(elements));
    
    for i = 1:length(elements)
        element = strtrim(elements{i});
        arr{i} = parse_json_basic(element);
    end
    
    % Convert to numeric array if all elements are numbers
    if all(cellfun(@isnumeric, arr))
        arr = cell2mat(arr);
    end
end

function str = parse_string(json_str)
% Parse JSON string
    if length(json_str) < 2 || json_str(1) ~= '"' || json_str(end) ~= '"'
        error('Invalid JSON string format');
    end
    
    str = json_str(2:end-1);
end

function val = parse_value(json_str)
% Parse JSON value (number, boolean, null)
    json_str = strtrim(json_str);
    
    if strcmp(json_str, 'null')
        val = [];
    elseif strcmp(json_str, 'true')
        val = true;
    elseif strcmp(json_str, 'false')
        val = false;
    else
        % Try to parse as number
        val = str2double(json_str);
        if isnan(val)
            % If not a number, treat as string
            val = json_str;
        end
    end
end

function parts = split_by_comma(str)
% Split string by comma, respecting nested structures
    parts = {};
    current = '';
    depth = 0;
    in_string = false;
    
    for i = 1:length(str)
        char = str(i);
        
        if char == '"' && (i == 1 || str(i-1) ~= '\')
            in_string = ~in_string;
        end
        
        if ~in_string
            if char == '{' || char == '['
                depth = depth + 1;
            elseif char == '}' || char == ']'
                depth = depth - 1;
            end
        end
        
        if char == ',' && depth == 0 && ~in_string
            parts{end+1} = current;
            current = '';
        else
            current = [current char];
        end
    end
    
    if ~isempty(current)
        parts{end+1} = current;
    end
end

function valid_name = makeValidName(name)
% Make a valid Octave variable name
% Simple implementation for Octave compatibility
    
    % Remove invalid characters and replace with underscore
    valid_name = regexprep(name, '[^a-zA-Z0-9_]', '_');
    
    % Ensure it starts with a letter
    if ~isempty(valid_name) && ~isstrprop(valid_name(1), 'alpha')
        valid_name = ['x' valid_name];
    end
    
    % Ensure it's not empty
    if isempty(valid_name)
        valid_name = 'x';
    end
end