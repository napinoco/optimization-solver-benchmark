function savejson(data, filename)
% SAVEJSON Save data to JSON file
%
% Usage:
%   savejson(data, filename)
%
% This function saves Octave/MATLAB data structures to a JSON file.
% Provides basic JSON encoding functionality.

    % Generate JSON string
    try
        if exist('jsonencode', 'file')
            % Use built-in JSON encoder if available (Octave 4.4+)
            json_str = jsonencode(data);
        else
            % Fallback to basic encoding
            json_str = encode_json_basic(data);
        end
    catch err
        error('Failed to encode JSON: %s', err.message);
    end
    
    % Write to file
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end
    
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

function json_str = encode_json_basic(data)
% Basic JSON encoder for simple structures

    if isstruct(data)
        json_str = encode_object(data);
    elseif iscell(data)
        json_str = encode_array(data);
    elseif ischar(data)
        json_str = encode_string(data);
    elseif isnumeric(data)
        json_str = encode_number(data);
    elseif islogical(data)
        json_str = encode_boolean(data);
    elseif isempty(data)
        json_str = 'null';
    else
        % Convert to string as fallback
        json_str = encode_string(char(data));
    end
end

function json_str = encode_object(obj)
% Encode struct as JSON object
    
    field_names = fieldnames(obj);
    pairs = {};
    
    for i = 1:length(field_names)
        field = field_names{i};
        value = obj.(field);
        
        key_str = encode_string(field);
        value_str = encode_json_basic(value);
        
        pairs{i} = sprintf('%s: %s', key_str, value_str);
    end
    
    if isempty(pairs)
        json_str = '{}';
    else
        json_str = sprintf('{%s}', strjoin(pairs, ', '));
    end
end

function json_str = encode_array(arr)
% Encode cell array as JSON array
    
    if isempty(arr)
        json_str = '[]';
        return;
    end
    
    elements = {};
    for i = 1:length(arr)
        elements{i} = encode_json_basic(arr{i});
    end
    
    json_str = sprintf('[%s]', strjoin(elements, ', '));
end

function json_str = encode_string(str)
% Encode string with proper escaping
    
    % Escape special characters
    str = strrep(str, '\', '\\');
    str = strrep(str, '"', '\"');
    str = strrep(str, sprintf('\n'), '\n');
    str = strrep(str, sprintf('\r'), '\r');
    str = strrep(str, sprintf('\t'), '\t');
    
    json_str = sprintf('"%s"', str);
end

function json_str = encode_number(num)
% Encode number (scalar or array)
    
    if isscalar(num)
        if isnan(num)
            json_str = 'null';
        elseif isinf(num)
            if num > 0
                json_str = 'null';  % JSON doesn't support Infinity
            else
                json_str = 'null';  % JSON doesn't support -Infinity
            end
        else
            json_str = sprintf('%.15g', num);
        end
    else
        % Handle arrays
        if isvector(num)
            elements = {};
            for i = 1:length(num)
                elements{i} = encode_number(num(i));
            end
            json_str = sprintf('[%s]', strjoin(elements, ', '));
        else
            % Multi-dimensional array - convert to nested arrays
            elements = {};
            for i = 1:size(num, 1)
                row_elements = {};
                for j = 1:size(num, 2)
                    row_elements{j} = encode_number(num(i, j));
                end
                elements{i} = sprintf('[%s]', strjoin(row_elements, ', '));
            end
            json_str = sprintf('[%s]', strjoin(elements, ', '));
        end
    end
end

function json_str = encode_boolean(bool)
% Encode boolean value
    
    if bool
        json_str = 'true';
    else
        json_str = 'false';
    end
end

function result = strjoin(strings, delimiter)
% Join cell array of strings with delimiter
% (Compatibility function for older Octave versions)
    
    if exist('strjoin', 'file')
        result = strjoin(strings, delimiter);
        return;
    end
    
    if isempty(strings)
        result = '';
        return;
    end
    
    result = strings{1};
    for i = 2:length(strings)
        result = [result delimiter strings{i}];
    end
end