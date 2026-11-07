classdef SignalDataLoader
    % SignalDataLoader - Universal experimental data import module
    %
    % Supports:
    %   - MATLAB binary files (.mat) [matrices, tables, timetables, or multi-channel vectors]
    %   - Delimited text files (.csv, .txt, .dat)
    %   - Excel spreadsheets (.xlsx, .xls)
    %
    % Key Features:
    %   - Automatic channel name extraction and sanitization
    %   - Robust time-column detection (via header matching or monotonic gradient test)
    %   - Automatic sampling frequency (Fs) estimation from timestamps
    %   - Flexible fallback for datasets without explicit time columns
    
    methods (Static)
        function [dataMatrix, channelNames, timeVector, detectedFs, isTimeDetected] = loadFile(filePath, firstColIsTimeOverride)
            % Load data from file and return standardized multi-channel arrays.
            %
            % Inputs:
            %   filePath               - Path to the data file (.mat, .csv, .xlsx, etc.)
            %   firstColIsTimeOverride - (Optional) boolean override:
            %                            true: treat column 1 as time
            %                            false: treat all columns as data channels
            %                            [] / omitted: auto-detect
            %
            % Outputs:
            %   dataMatrix     - N-by-C matrix of channel measurements (double)
            %   channelNames   - 1-by-C cell array of string names for channels
            %   timeVector     - N-by-1 vector of time values (empty if no time column)
            %   detectedFs     - Estimated sampling frequency in Hz (NaN if cannot determine)
            %   isTimeDetected - Logical flag indicating whether column 1 was treated as time
            
            if nargin < 2
                firstColIsTimeOverride = [];
            end
            
            if ~exist(filePath, 'file')
                error('SignalDataLoader:FileNotFound', 'Specified file does not exist: %s', filePath);
            end
            
            [~, ~, ext] = fileparts(filePath);
            ext = lower(ext);
            
            switch ext
                case '.mat'
                    [rawMatrix, rawNames] = SignalDataLoader.loadMatFile(filePath);
                case {'.csv', '.txt', '.dat'}
                    [rawMatrix, rawNames] = SignalDataLoader.loadTextFile(filePath);
                case {'.xlsx', '.xls'}
                    [rawMatrix, rawNames] = SignalDataLoader.loadExcelFile(filePath);
                otherwise
                    error('SignalDataLoader:UnsupportedFormat', ...
                          'Unsupported file extension: %s. Supported formats: .mat, .csv, .txt, .xlsx, .xls', ext);
            end
            
            % Validate extracted raw data
            if isempty(rawMatrix) || ~isnumeric(rawMatrix)
                error('SignalDataLoader:EmptyData', 'No numeric data could be extracted from: %s', filePath);
            end
            
            % Ensure rows = time steps, cols = channels (transpose if row vector)
            if size(rawMatrix, 1) == 1 && size(rawMatrix, 2) > 1
                rawMatrix = rawMatrix';
            end
            
            numCols = size(rawMatrix, 2);
            numRows = size(rawMatrix, 1);
            
            if isempty(rawNames) || length(rawNames) ~= numCols
                rawNames = cell(1, numCols);
                for c = 1:numCols
                    rawNames{c} = sprintf('Channel %d', c);
                end
            end
            
            % Determine if column 1 should be treated as time
            isTimeDetected = false;
            col1 = rawMatrix(:, 1);
            
            if isempty(firstColIsTimeOverride)
                % Automatic detection logic
                col1Name = lower(strtrim(rawNames{1}));
                hasTimeKeyword = any(contains(col1Name, {'time', 't_sec', 't (s)', 't(s)', 'seconds', 'timestamp'}));
                
                % Check if column 1 is strictly monotonically increasing with regular steps
                diffs = diff(col1);
                isMonotonic = all(diffs > 0);
                isUniform = false;
                if isMonotonic && length(diffs) > 1
                    cv = std(diffs) / mean(diffs);
                    isUniform = (cv < 0.05); % less than 5% jitter
                end
                
                if (hasTimeKeyword || (isMonotonic && isUniform)) && numCols > 1
                    isTimeDetected = true;
                end
            else
                isTimeDetected = (firstColIsTimeOverride && numCols > 1);
            end
            
            if isTimeDetected
                timeVector = col1;
                dataMatrix = rawMatrix(:, 2:end);
                channelNames = rawNames(2:end);
                
                % Estimate Fs from time difference
                dt = median(diff(timeVector));
                if dt > 0
                    detectedFs = round(1.0 / dt, 6);
                else
                    detectedFs = NaN;
                end
            else
                timeVector = [];
                dataMatrix = rawMatrix;
                channelNames = rawNames;
                detectedFs = NaN;
            end
        end
        
        function [rawMatrix, rawNames] = loadMatFile(filePath)
            % Specialized .mat loader handling matrices, tables, or channel vectors
            S = load(filePath);
            fields = fieldnames(S);
            
            if isempty(fields)
                error('SignalDataLoader:EmptyMatFile', 'MAT file contains no variables.');
            end
            
            % Step 1: Collect numeric vector, 2D matrix, and table variables
            numericVectorFields = {};
            numericMatrixFields = {};
            tableFields = {};
            
            for i = 1:length(fields)
                fn = fields{i};
                val = S.(fn);
                if istable(val) || istimetable(val)
                    tableFields{end+1} = fn; %#ok<AGROW>
                elseif isnumeric(val)
                    if isvector(val) && numel(val) > 1
                        numericVectorFields{end+1} = fn; %#ok<AGROW>
                    elseif ismatrix(val) && size(val, 1) > 1 && size(val, 2) > 1
                        numericMatrixFields{end+1} = fn; %#ok<AGROW>
                    end
                end
            end
            
            % Case 1: Multiple numeric vectors of equal length
            if length(numericVectorFields) > 1
                % Prioritize vectors matching channel or data naming pattern
                patFields = numericVectorFields(contains(numericVectorFields, '_Data') | ...
                                                contains(numericVectorFields, 'channel', 'IgnoreCase', true));
                if length(patFields) > 1
                    candidateFields = patFields;
                else
                    candidateFields = numericVectorFields;
                end
                
                lengths = cellfun(@(f) numel(S.(f)), candidateFields);
                if all(lengths == lengths(1)) && lengths(1) > 1
                    [sortedFields, sortOrder] = SignalDataLoader.naturalSort(candidateFields);
                    N = lengths(1);
                    C = length(sortedFields);
                    rawMatrix = zeros(N, C);
                    rawNames = cell(1, C);
                    for c = 1:C
                        % Use sortedFields{c} (== candidateFields{sortOrder(c)}) to fetch data
                        v = S.(sortedFields{c});
                        rawMatrix(:, c) = v(:);
                        cleanName = regexprep(sortedFields{c}, '(_Data|_data)$', '');
                        rawNames{c} = strrep(cleanName, '_', ' ');
                    end
                    return;
                end
            end
            
            % Case 2: Tables or Timetables
            if ~isempty(tableFields)
                val = S.(tableFields{1});
                if istimetable(val)
                    t = seconds(val.Properties.RowTimes);
                    d = val{:, :};
                    rawMatrix = [t, d];
                    rawNames = [{'Time'}, val.Properties.VariableNames];
                else
                    rawMatrix = val{:, :};
                    rawNames = val.Properties.VariableNames;
                end
                for c = 1:length(rawNames)
                    rawNames{c} = strrep(rawNames{c}, '_', ' ');
                end
                return;
            end
            
            % Case 3: 2D Numeric Matrix with multiple columns
            if ~isempty(numericMatrixFields)
                fn = numericMatrixFields{1};
                rawMatrix = S.(fn);
                C = size(rawMatrix, 2);
                rawNames = cell(1, C);
                for c = 1:C
                    rawNames{c} = sprintf('%s Col %d', strrep(fn, '_', ' '), c);
                end
                return;
            end
            
            % Case 4: Single numeric vector fallback
            if length(numericVectorFields) == 1
                fn = numericVectorFields{1};
                val = S.(fn);
                rawMatrix = val(:);
                cleanName = regexprep(fn, '(_Data|_data)$', '');
                rawNames = {strrep(cleanName, '_', ' ')};
                return;
            end
            
            % Fallback: take first numeric variable
            for i = 1:length(fields)
                val = S.(fields{i});
                if isnumeric(val)
                    rawMatrix = val;
                    if size(rawMatrix, 1) == 1 && size(rawMatrix, 2) > 1
                        rawMatrix = rawMatrix';
                    end
                    C = size(rawMatrix, 2);
                    rawNames = cell(1, C);
                    for c = 1:C
                        rawNames{c} = sprintf('%s Col %d', strrep(fields{i}, '_', ' '), c);
                    end
                    return;
                end
            end
            
            error('SignalDataLoader:MatParsingError', 'Could not locate numeric signal arrays in MAT file.');
        end
        
        function [rawMatrix, rawNames] = loadTextFile(filePath)
            % Import Delimited Text/CSV
            try
                opts = detectImportOptions(filePath);
                T = readtable(filePath, opts);
                rawNames = T.Properties.VariableNames;
                
                % Convert table to numeric array
                isNumCol = varfun(@isnumeric, T, 'OutputFormat', 'uniform');
                if ~all(isNumCol)
                    T = T(:, isNumCol);
                    rawNames = rawNames(isNumCol);
                end
                rawMatrix = table2array(T);
            catch
                % Fallback to readmatrix
                try
                    rawMatrix = readmatrix(filePath);
                    rawNames = {};
                catch
                    rawMatrix = dlmread(filePath); %#ok<DLMRD>
                    rawNames = {};
                end
            end
            
            % Clean up rawNames formatting
            for i = 1:length(rawNames)
                rawNames{i} = strrep(rawNames{i}, '_', ' ');
            end
        end
        
        function [rawMatrix, rawNames] = loadExcelFile(filePath)
            % Import Excel spreadsheets
            try
                opts = detectImportOptions(filePath);
                T = readtable(filePath, opts);
                rawNames = T.Properties.VariableNames;
                isNumCol = varfun(@isnumeric, T, 'OutputFormat', 'uniform');
                if ~all(isNumCol)
                    T = T(:, isNumCol);
                    rawNames = rawNames(isNumCol);
                end
                rawMatrix = table2array(T);
            catch
                try
                    rawMatrix = readmatrix(filePath);
                    rawNames = {};
                catch
                    [rawMatrix, ~, ~] = xlsread(filePath); %#ok<XLSRD>
                    rawNames = {};
                end
            end
            
            for i = 1:length(rawNames)
                rawNames{i} = strrep(rawNames{i}, '_', ' ');
            end
        end
        
        function [sortedList, order] = naturalSort(charList)
            % Natural sorting helper for strings like 'Channel_1', 'Channel_2', 'Channel_10'
            % Returns:
            %   sortedList - cell array of strings in natural numeric order
            %   order      - permutation index such that sortedList = charList(order)
            try
                % Extract the last numeric substring from each name and sort by it
                nums = zeros(1, length(charList));
                for k = 1:length(charList)
                    tokens = regexp(charList{k}, '\d+', 'match');
                    if ~isempty(tokens)
                        nums(k) = str2double(tokens{end});
                    else
                        nums(k) = k;  % fall back to positional index for non-numeric names
                    end
                end
                [~, order] = sort(nums);
                sortedList = charList(order);
            catch
                % Fallback: lexicographic sort; still return both outputs
                [sortedList, order] = sort(charList);
            end
        end
    end
end

