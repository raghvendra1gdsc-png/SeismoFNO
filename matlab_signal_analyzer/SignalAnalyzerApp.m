classdef SignalAnalyzerApp < handle
    % SignalAnalyzerApp - Research-Grade Multi-Channel Signal & Spectral Analyzer
    %
    % Interactive MATLAB application for multi-channel experimental data analysis.
    % Integrates universal ingestion (.mat, .xlsx, .csv), high-resolution time
    % domain visualization, and single-sided Fast Fourier Transform (FFT) with
    % windowing, coherent gain compensation, and automated modal peak picking.
    %
    % Launch commands:
    %   SignalAnalyzerApp                          % Launches GUI with default sample prompt
    %   SignalAnalyzerApp('path/to/datafile.mat')   % Directly opens and visualizes file
    
    properties (Access = public)
        % UI Components
        UIFigure             matlab.ui.Figure
        MainGrid             matlab.ui.container.GridLayout
        SidebarPanel         matlab.ui.container.Panel
        SidebarGrid          matlab.ui.container.GridLayout
        DisplayPanel         matlab.ui.container.Panel
        DisplayGrid          matlab.ui.container.GridLayout
        
        % Axes
        TimeAxes             matlab.ui.control.UIAxes
        FreqAxes             matlab.ui.control.UIAxes
        
        % Sidebar Controls - File
        FileButton           matlab.ui.control.Button
        FileLabel            matlab.ui.control.Label
        FirstColTimeCheck    matlab.ui.control.CheckBox
        FsEditField          matlab.ui.control.NumericEditField
        AutoFsButton         matlab.ui.control.Button
        InfoLabel            matlab.ui.control.Label
        
        % Sidebar Controls - Channels
        ChannelListBox       matlab.ui.control.ListBox
        SelectAllButton      matlab.ui.control.Button
        ClearAllButton       matlab.ui.control.Button
        
        % Sidebar Controls - DSP Configuration
        DetrendDropDown      matlab.ui.control.DropDown
        WindowDropDown       matlab.ui.control.DropDown
        NfftDropDown         matlab.ui.control.DropDown
        ScaleDropDown        matlab.ui.control.DropDown
        MaxFreqEditField     matlab.ui.control.NumericEditField
        
        % Sidebar Controls - Peak Picking
        PeakCheck            matlab.ui.control.CheckBox
        NumPeaksSpinner      matlab.ui.control.Spinner
        
        % Sidebar Controls - Export
        ExportFigButton      matlab.ui.control.Button
        ExportCSVButton      matlab.ui.control.Button
        
        % Status Strip
        StatusLabel          matlab.ui.control.Label
    end
    
    properties (Access = private)
        % Core Data State
        CurrentFilePath      char = ''
        RawMatrix            double = []
        RawChannelNames      cell = {}
        DataMatrix           double = []
        ChannelNames         cell = {}
        TimeVector           double = []
        SamplingFreq         double = 200.0
        IsTimeDetected       logical = false
        
        % Precomputed Spectra Cache
        ComputedFreqs        cell = {}
        ComputedAmps         cell = {}
        ComputeddB           cell = {}
        ComputedPSD          cell = {}
        DetectedPeaks        cell = {}
        
        % Plot Handles
        TimePlotLines        = []
        FreqPlotLines        = []
        PeakPlotMarkers      = []
        PeakPlotTexts        = []
        
        % Color Palette
        ColorPalette = [
            0.0000, 0.4470, 0.7410;  % Blue
            0.8500, 0.3250, 0.0980;  % Orange-Red
            0.9290, 0.6940, 0.1250;  % Yellow-Gold
            0.4940, 0.1840, 0.5560;  % Purple
            0.4660, 0.6740, 0.1880;  % Green
            0.3010, 0.7450, 0.9330;  % Light Blue
            0.6350, 0.0780, 0.1840;  % Dark Red
            0.2000, 0.2000, 0.2000;  % Charcoal
        ];
    end
    
    methods (Access = public)
        function app = SignalAnalyzerApp(initialFile)
            % Ensure dependencies in src/ are accessible
            appDir = fileparts(mfilename('fullpath'));
            srcDir = fullfile(appDir, 'src');
            if exist(srcDir, 'dir')
                addpath(srcDir);
            end
            
            % Construct GUI
            app.buildUI();
            
            % If initial file provided, load it; otherwise check for sample data
            if nargin >= 1 && ~isempty(initialFile) && exist(initialFile, 'file')
                app.loadFile(initialFile);
            else
                % Check for default NonClassical_200Hz file
                sampleMat = fullfile(appDir, 'sample_data', 'NonClassical_200Hz.mat');
                sampleCsv = fullfile(appDir, 'sample_data', 'NonClassical_200Hz.csv');
                if exist(sampleMat, 'file')
                    app.loadFile(sampleMat);
                elseif exist(sampleCsv, 'file')
                    app.loadFile(sampleCsv);
                end
            end
        end
        
        function loadFile(app, filePath)
            % Load dataset using SignalDataLoader
            try
                app.StatusLabel.Text = sprintf('Loading %s...', filePath);
                drawnow;
                
                overrideTime = [];
                [app.DataMatrix, app.ChannelNames, app.TimeVector, detectedFs, app.IsTimeDetected] = ...
                    SignalDataLoader.loadFile(filePath, overrideTime);
                
                app.CurrentFilePath = filePath;
                [~, fname, fext] = fileparts(filePath);
                app.FileLabel.Text = sprintf('%s%s (%d samples, %d channels)', fname, fext, size(app.DataMatrix, 1), size(app.DataMatrix, 2));
                
                % Update Time checkbox
                app.FirstColTimeCheck.Value = app.IsTimeDetected;
                
                % Update Fs
                if ~isnan(detectedFs) && detectedFs > 0
                    app.SamplingFreq = detectedFs;
                elseif isempty(app.SamplingFreq) || app.SamplingFreq <= 0
                    app.SamplingFreq = 200.0;
                end
                app.FsEditField.Value = app.SamplingFreq;
                if app.IsTimeDetected
                    app.FsEditField.Enable = 'off';
                    app.AutoFsButton.Enable = 'on';
                else
                    app.FsEditField.Enable = 'on';
                    app.AutoFsButton.Enable = 'off';
                end
                
                % If no time vector, construct one
                if isempty(app.TimeVector)
                    N = size(app.DataMatrix, 1);
                    app.TimeVector = (0:N-1)' / app.SamplingFreq;
                end
                
                % Update Channel ListBox
                app.ChannelListBox.Items = app.ChannelNames;
                if ~isempty(app.ChannelNames)
                    % By default select all channels
                    app.ChannelListBox.Value = app.ChannelNames;
                end
                
                % Update Info Label
                duration = size(app.DataMatrix, 1) / app.SamplingFreq;
                app.InfoLabel.Text = sprintf('Duration: %.2f s | Fs: %.1f Hz | Nyquist: %.1f Hz', ...
                                             duration, app.SamplingFreq, app.SamplingFreq/2);
                
                % Recompute and Update Plots
                app.updateAnalysis();
                app.StatusLabel.Text = sprintf('Loaded %s successfully.', fname);
            catch ME
                errordlg(sprintf('Failed to load file:\n%s', ME.message), 'Data Import Error');
                app.StatusLabel.Text = 'Error loading file.';
            end
        end
        
        function updateAnalysis(app)
            % Recompute FFT and refresh all visualizations
            if isempty(app.DataMatrix)
                return;
            end
            
            selectedChannels = app.ChannelListBox.Value;
            if isempty(selectedChannels)
                cla(app.TimeAxes);
                cla(app.FreqAxes);
                app.StatusLabel.Text = 'No channels selected.';
                return;
            end
            
            % Map channel names to indices
            chIndices = [];
            for i = 1:length(selectedChannels)
                match = find(strcmp(app.ChannelNames, selectedChannels{i}));
                if ~isempty(match)
                    chIndices(end+1) = match(1); %#ok<AGROW>
                end
            end
            
            if isempty(chIndices)
                return;
            end
            
            % Retrieve DSP Options
            Fs = app.FsEditField.Value;
            if Fs <= 0, Fs = 200.0; end
            app.SamplingFreq = Fs;
            
            % Update time vector if length doesn't match
            N = size(app.DataMatrix, 1);
            if isempty(app.TimeVector) || length(app.TimeVector) ~= N
                app.TimeVector = (0:N-1)' / Fs;
            end
            
            % DSP options struct
            dspOpts = struct();
            switch app.DetrendDropDown.Value
                case 'DC Offset / Mean'
                    dspOpts.detrendMethod = 'mean';
                case 'Linear Trend'
                    dspOpts.detrendMethod = 'linear';
                case 'None (Raw)'
                    dspOpts.detrendMethod = 'none';
            end
            
            switch app.WindowDropDown.Value
                case 'Hann (Hanning)'
                    dspOpts.windowType = 'hann';
                case 'Hamming'
                    dspOpts.windowType = 'hamming';
                case 'Blackman'
                    dspOpts.windowType = 'blackman';
                case 'Flat Top'
                    dspOpts.windowType = 'flattop';
                case 'Rectangular (None)'
                    dspOpts.windowType = 'rectangular';
            end
            
            switch app.NfftDropDown.Value
                case 'Next Power of 2'
                    dspOpts.nfft = 2^nextpow2(N);
                case 'Original N'
                    dspOpts.nfft = N;
                case '2x Zero-Padded'
                    dspOpts.nfft = 2 * (2^nextpow2(N));
                case '4x Zero-Padded'
                    dspOpts.nfft = 4 * (2^nextpow2(N));
            end
            
            % Compute FFT for all selected channels
            numSel = length(chIndices);
            app.ComputedFreqs = cell(1, numSel);
            app.ComputedAmps  = cell(1, numSel);
            app.ComputeddB    = cell(1, numSel);
            app.ComputedPSD   = cell(1, numSel);
            app.DetectedPeaks = cell(1, numSel);
            
            numPeaksToFind = app.NumPeaksSpinner.Value;
            
            for k = 1:numSel
                col = chIndices(k);
                rawSignal = app.DataMatrix(:, col);
                
                [f, amp, psd, amp_dB] = SignalProcessor.computeFFT(rawSignal, Fs, dspOpts);
                
                app.ComputedFreqs{k} = f;
                app.ComputedAmps{k}  = amp;
                app.ComputeddB{k}    = amp_dB;
                app.ComputedPSD{k}   = psd;
                
                if app.PeakCheck.Value
                    peaks = SignalProcessor.findProminentPeaks(f, amp, numPeaksToFind, 0.05, 0.1, 0.5);
                    app.DetectedPeaks{k} = peaks;
                else
                    app.DetectedPeaks{k} = [];
                end
            end
            
            % Refresh Plots
            app.renderTimeDomainPlot(chIndices, dspOpts);
            app.renderFreqDomainPlot(chIndices);
            
            % Update Status bar with dominant frequencies
            app.updateSummaryStatus(chIndices);
        end
    end
    
    methods (Access = private)
        function buildUI(app)
            % Construct interactive user interface
            screenSize = get(0, 'ScreenSize');
            figWidth = min(1280, screenSize(3) - 80);
            figHeight = min(820, screenSize(4) - 100);
            
            app.UIFigure = uifigure('Name', 'Multi-Channel Experimental Signal & Spectral Analyzer', ...
                                    'Position', [50, 50, figWidth, figHeight], ...
                                    'Color', [0.96, 0.96, 0.97]);
            
            % Top-level Grid: 1 Row, 2 Columns (Sidebar: 340px, Display: remaining)
            app.MainGrid = uigridlayout(app.UIFigure, [1, 2]);
            app.MainGrid.ColumnWidth = {340, '1x'};
            app.MainGrid.RowHeight = {'1x'};
            app.MainGrid.Padding = [10, 10, 10, 10];
            app.MainGrid.ColumnSpacing = 10;
            
            % ---------------- SIDEBAR ----------------
            app.SidebarPanel = uipanel(app.MainGrid, 'Title', 'Configuration & Analysis Control', ...
                                      'FontWeight', 'bold', 'FontSize', 12, ...
                                      'BackgroundColor', [1, 1, 1], 'Scrollable', 'on');
            app.SidebarGrid = uigridlayout(app.SidebarPanel, [18, 1]);
            app.SidebarGrid.RowHeight = {32, 22, 25, 28, 25, 18, 110, 28, 18, 28, 28, 28, 28, 28, 25, 28, 32, 32};
            app.SidebarGrid.Padding = [8, 8, 8, 8];
            app.SidebarGrid.RowSpacing = 6;
            
            % Row 1: File Button
            app.FileButton = uibutton(app.SidebarGrid, 'push', 'Text', '📁 Load File (.mat, .csv, .xlsx)', ...
                                      'FontWeight', 'bold', 'BackgroundColor', [0.12, 0.44, 0.71], ...
                                      'FontColor', [1, 1, 1], ...
                                      'ButtonPushedFcn', @(~, ~) app.onSelectFilePushed());
            
            % Row 2: File Label
            app.FileLabel = uilabel(app.SidebarGrid, 'Text', 'No file loaded', 'FontAngle', 'italic', ...
                                    'FontColor', [0.3, 0.3, 0.3]);
            
            % Row 3: First Column is Time Checkbox
            app.FirstColTimeCheck = uicheckbox(app.SidebarGrid, 'Text', 'First column is Time vector', ...
                                              'Value', true, ...
                                              'ValueChangedFcn', @(~, ~) app.onTimeCheckChanged());
            
            % Row 4: Sampling Frequency Grid
            fsSubGrid = uigridlayout(app.SidebarGrid, [1, 3]);
            fsSubGrid.ColumnWidth = {'fit', '1x', 'fit'};
            fsSubGrid.Padding = [0, 0, 0, 0];
            uilabel(fsSubGrid, 'Text', 'Fs (Hz):', 'FontWeight', 'bold');
            app.FsEditField = uieditfield(fsSubGrid, 'numeric', 'Value', 200.0, ...
                                          'Limits', [0.001, 1e9], ...
                                          'ValueChangedFcn', @(~, ~) app.onFsChanged());
            app.AutoFsButton = uibutton(fsSubGrid, 'push', 'Text', 'Auto Δt', ...
                                        'Tooltip', 'Calculate Fs from timestamp difference', ...
                                        'ButtonPushedFcn', @(~, ~) app.onAutoFsPushed());
            
            % Row 5: Metadata Info
            app.InfoLabel = uilabel(app.SidebarGrid, 'Text', 'Duration: -- | Fs: --', ...
                                    'FontSize', 10, 'FontColor', [0.4, 0.4, 0.4]);
            
            % Row 6: Section Header: Channels
            uilabel(app.SidebarGrid, 'Text', 'SELECT CHANNELS TO ANALYZE:', 'FontWeight', 'bold', ...
                    'FontSize', 10, 'FontColor', [0.2, 0.2, 0.2]);
            
            % Row 7: Channel ListBox
            app.ChannelListBox = uilistbox(app.SidebarGrid, 'Items', {}, 'Multiselect', 'on', ...
                                           'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 8: Select All / Clear All
            btnGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            btnGrid.ColumnWidth = {'1x', '1x'};
            btnGrid.Padding = [0, 0, 0, 0];
            app.SelectAllButton = uibutton(btnGrid, 'push', 'Text', 'Select All', ...
                                           'ButtonPushedFcn', @(~, ~) app.onSelectAll());
            app.ClearAllButton = uibutton(btnGrid, 'push', 'Text', 'Clear All', ...
                                          'ButtonPushedFcn', @(~, ~) app.onClearAll());
            
            % Row 9: Section Header: DSP
            uilabel(app.SidebarGrid, 'Text', 'SIGNAL PROCESSING & FFT SETTINGS:', 'FontWeight', 'bold', ...
                    'FontSize', 10, 'FontColor', [0.2, 0.2, 0.2]);
            
            % Row 10: Detrending Dropdown
            detrendGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            detrendGrid.ColumnWidth = {110, '1x'};
            detrendGrid.Padding = [0, 0, 0, 0];
            uilabel(detrendGrid, 'Text', 'Detrending:');
            app.DetrendDropDown = uidropdown(detrendGrid, 'Items', {'DC Offset / Mean', 'Linear Trend', 'None (Raw)'}, ...
                                             'Value', 'DC Offset / Mean', ...
                                             'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 11: Window Dropdown
            winGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            winGrid.ColumnWidth = {110, '1x'};
            winGrid.Padding = [0, 0, 0, 0];
            uilabel(winGrid, 'Text', 'Window Function:');
            app.WindowDropDown = uidropdown(winGrid, 'Items', {'Hann (Hanning)', 'Hamming', 'Blackman', 'Flat Top', 'Rectangular (None)'}, ...
                                           'Value', 'Hann (Hanning)', ...
                                           'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 12: FFT Length / Zero-Padding
            nfftGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            nfftGrid.ColumnWidth = {110, '1x'};
            nfftGrid.Padding = [0, 0, 0, 0];
            uilabel(nfftGrid, 'Text', 'FFT Resolution:');
            app.NfftDropDown = uidropdown(nfftGrid, 'Items', {'Next Power of 2', 'Original N', '2x Zero-Padded', '4x Zero-Padded'}, ...
                                          'Value', 'Next Power of 2', ...
                                          'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 13: Spectral Vertical Scale
            scaleGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            scaleGrid.ColumnWidth = {110, '1x'};
            scaleGrid.Padding = [0, 0, 0, 0];
            uilabel(scaleGrid, 'Text', 'Vertical Scale:');
            app.ScaleDropDown = uidropdown(scaleGrid, 'Items', {'Linear Amplitude', 'Decibels (dB)', 'PSD (Power Spectrum)'}, ...
                                           'Value', 'Linear Amplitude', ...
                                           'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 14: Frequency Max Display
            freqLimGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            freqLimGrid.ColumnWidth = {110, '1x'};
            freqLimGrid.Padding = [0, 0, 0, 0];
            uilabel(freqLimGrid, 'Text', 'Max Freq [Hz]:');
            app.MaxFreqEditField = uieditfield(freqLimGrid, 'numeric', 'Value', 0, ...
                                               'Tooltip', 'Set 0 to display up to Nyquist (Fs/2)', ...
                                               'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 15: Modal Peak Detection
            peakGrid = uigridlayout(app.SidebarGrid, [1, 3]);
            peakGrid.ColumnWidth = {'fit', 'fit', '1x'};
            peakGrid.Padding = [0, 0, 0, 0];
            app.PeakCheck = uicheckbox(peakGrid, 'Text', 'Label Peaks', 'Value', true, ...
                                       'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            uilabel(peakGrid, 'Text', 'Top:');
            app.NumPeaksSpinner = uispinner(peakGrid, 'Limits', [1, 10], 'Value', 3, ...
                                            'ValueChangedFcn', @(~, ~) app.updateAnalysis());
            
            % Row 16: Export Buttons
            exportGrid = uigridlayout(app.SidebarGrid, [1, 2]);
            exportGrid.ColumnWidth = {'1x', '1x'};
            exportGrid.Padding = [0, 0, 0, 0];
            app.ExportFigButton = uibutton(exportGrid, 'push', 'Text', '📷 Export Plot', ...
                                           'ButtonPushedFcn', @(~, ~) app.onExportPlot());
            app.ExportCSVButton = uibutton(exportGrid, 'push', 'Text', '💾 Export FFT', ...
                                           'ButtonPushedFcn', @(~, ~) app.onExportCSV());
            
            % ---------------- MAIN DISPLAY AREA ----------------
            app.DisplayPanel = uipanel(app.MainGrid, 'Title', 'Interactive Response Visualization', ...
                                      'FontWeight', 'bold', 'FontSize', 12, ...
                                      'BackgroundColor', [1, 1, 1]);
            app.DisplayGrid = uigridlayout(app.DisplayPanel, [3, 1]);
            app.DisplayGrid.RowHeight = {'1x', '1x', 30};
            app.DisplayGrid.Padding = [10, 10, 10, 10];
            app.DisplayGrid.RowSpacing = 10;
            
            % Time Domain Axes
            app.TimeAxes = uiaxes(app.DisplayGrid);
            title(app.TimeAxes, 'Time-Domain Dynamic Response', 'FontWeight', 'bold');
            xlabel(app.TimeAxes, 'Time (seconds)', 'FontWeight', 'bold');
            ylabel(app.TimeAxes, 'Amplitude', 'FontWeight', 'bold');
            grid(app.TimeAxes, 'on');
            box(app.TimeAxes, 'on');
            app.TimeAxes.XGrid = 'on'; app.TimeAxes.YGrid = 'on';
            app.TimeAxes.GridColor = [0.85, 0.85, 0.85];
            app.TimeAxes.GridAlpha = 0.8;
            
            % Frequency Domain Axes
            app.FreqAxes = uiaxes(app.DisplayGrid);
            title(app.FreqAxes, 'Frequency-Domain Spectrum (FFT)', 'FontWeight', 'bold');
            xlabel(app.FreqAxes, 'Frequency (Hz)', 'FontWeight', 'bold');
            ylabel(app.FreqAxes, 'Single-Sided Amplitude', 'FontWeight', 'bold');
            grid(app.FreqAxes, 'on');
            box(app.FreqAxes, 'on');
            app.FreqAxes.XGrid = 'on'; app.FreqAxes.YGrid = 'on';
            app.FreqAxes.GridColor = [0.85, 0.85, 0.85];
            app.FreqAxes.GridAlpha = 0.8;
            
            % Status Bar
            app.StatusLabel = uilabel(app.DisplayGrid, 'Text', 'Ready. Load a dataset to begin.', ...
                                      'FontSize', 11, 'FontColor', [0.2, 0.2, 0.2]);
        end
        
        function renderTimeDomainPlot(app, chIndices, dspOpts)
            % Plot selected channels in time domain
            cla(app.TimeAxes);
            hold(app.TimeAxes, 'on');
            
            t = app.TimeVector;
            numSel = length(chIndices);
            numColors = size(app.ColorPalette, 1);
            legendEntries = cell(1, numSel);
            app.TimePlotLines = gobjects(1, numSel);
            
            detrendMode = 'none';
            if nargin >= 3 && isfield(dspOpts, 'detrendMethod')
                detrendMode = dspOpts.detrendMethod;
            end
            
            for k = 1:numSel
                col = chIndices(k);
                y = app.DataMatrix(:, col);
                y_plot = SignalProcessor.detrendSignal(y, detrendMode);
                c = app.ColorPalette(mod(col-1, numColors) + 1, :);
                
                app.TimePlotLines(k) = plot(app.TimeAxes, t, y_plot, 'LineWidth', 1.2, 'Color', c);
                legendEntries{k} = app.ChannelNames{col};
            end
            
            hold(app.TimeAxes, 'off');
            
            % Update title and ylabel to reflect active detrending
            detrendVal = app.DetrendDropDown.Value;
            if strcmp(detrendVal, 'None (Raw)')
                ylabel(app.TimeAxes, 'Amplitude (Raw)', 'FontWeight', 'bold');
                title(app.TimeAxes, 'Time-Domain Dynamic Response (Raw)', 'FontWeight', 'bold');
            else
                ylabel(app.TimeAxes, sprintf('Amplitude (%s)', detrendVal), 'FontWeight', 'bold');
                title(app.TimeAxes, sprintf('Time-Domain Dynamic Response (%s)', detrendVal), 'FontWeight', 'bold');
            end
            
            if numSel > 0
                lgd = legend(app.TimeAxes, legendEntries, 'Location', 'northeast', 'AutoUpdate', 'off');
                lgd.ItemHitFcn = @(src, evt) app.onLegendHit(src, evt);
            end
            xlim(app.TimeAxes, [t(1), t(end)]);
        end
        
        function renderFreqDomainPlot(app, chIndices)
            % Plot computed spectra in frequency domain
            cla(app.FreqAxes);
            hold(app.FreqAxes, 'on');
            
            numSel = length(chIndices);
            numColors = size(app.ColorPalette, 1);
            legendEntries = cell(1, numSel);
            
            app.FreqPlotLines = gobjects(1, numSel);
            app.PeakPlotMarkers = cell(1, numSel);
            app.PeakPlotTexts = cell(1, numSel);
            
            scaleMode = app.ScaleDropDown.Value;
            switch scaleMode
                case 'Decibels (dB)'
                    ylabel(app.FreqAxes, 'Magnitude (dB)', 'FontWeight', 'bold');
                case 'PSD (Power Spectrum)'
                    ylabel(app.FreqAxes, 'Power Spectral Density (Units^2 / Hz)', 'FontWeight', 'bold');
                otherwise
                    ylabel(app.FreqAxes, 'Single-Sided Linear Amplitude', 'FontWeight', 'bold');
            end
            
            maxFreqToShow = app.MaxFreqEditField.Value;
            if maxFreqToShow <= 0
                maxFreqToShow = app.SamplingFreq / 2;
            end
            
            for k = 1:numSel
                col = chIndices(k);
                f = app.ComputedFreqs{k};
                
                switch scaleMode
                    case 'Decibels (dB)'
                        spec = app.ComputeddB{k};
                    case 'PSD (Power Spectrum)'
                        spec = app.ComputedPSD{k};
                    otherwise
                        spec = app.ComputedAmps{k};
                end
                
                c = app.ColorPalette(mod(col-1, numColors) + 1, :);
                app.FreqPlotLines(k) = plot(app.FreqAxes, f, spec, 'LineWidth', 1.4, 'Color', c);
                legendEntries{k} = app.ChannelNames{col};
                
                % Annotate Peaks
                markers = [];
                labels = [];
                if app.PeakCheck.Value && ~isempty(app.DetectedPeaks{k})
                    peaks = app.DetectedPeaks{k};
                    for p = 1:length(peaks)
                        pkFreq = peaks(p).frequency;
                        if pkFreq <= maxFreqToShow
                            [~, pIdx] = min(abs(f - pkFreq));
                            pkVal = spec(pIdx);
                            
                            hM = plot(app.FreqAxes, pkFreq, pkVal, 'v', 'MarkerSize', 7, ...
                                      'MarkerEdgeColor', 'k', 'MarkerFaceColor', c);
                            markers(end+1) = hM; %#ok<AGROW>
                            
                            labelStr = sprintf(' %.2f Hz', pkFreq);
                            hT = text(app.FreqAxes, pkFreq, pkVal, labelStr, ...
                                      'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.1], ...
                                      'VerticalAlignment', 'bottom');
                            labels(end+1) = hT; %#ok<AGROW>
                        end
                    end
                end
                app.PeakPlotMarkers{k} = markers;
                app.PeakPlotTexts{k} = labels;
            end
            
            hold(app.FreqAxes, 'off');
            if numSel > 0
                lgd = legend(app.FreqAxes, legendEntries, 'Location', 'northeast', 'AutoUpdate', 'off');
                lgd.ItemHitFcn = @(src, evt) app.onLegendHit(src, evt);
            end
            xlim(app.FreqAxes, [0, maxFreqToShow]);
        end
        
        function onLegendHit(app, ~, event)
            % Linked legend visibility toggle between Time and Frequency domains
            hitPeer = event.Peer;
            if isempty(hitPeer) || ~isvalid(hitPeer)
                return;
            end
            
            if strcmp(hitPeer.Visible, 'on')
                newVis = 'off';
            else
                newVis = 'on';
            end
            
            % Identify channel index k corresponding to clicked line
            k = find(app.TimePlotLines == hitPeer, 1);
            if isempty(k)
                k = find(app.FreqPlotLines == hitPeer, 1);
            end
            
            if ~isempty(k)
                if k <= length(app.TimePlotLines) && isvalid(app.TimePlotLines(k))
                    app.TimePlotLines(k).Visible = newVis;
                end
                if k <= length(app.FreqPlotLines) && isvalid(app.FreqPlotLines(k))
                    app.FreqPlotLines(k).Visible = newVis;
                end
                if k <= length(app.PeakPlotMarkers) && ~isempty(app.PeakPlotMarkers{k})
                    for m = 1:length(app.PeakPlotMarkers{k})
                        if isvalid(app.PeakPlotMarkers{k}(m))
                            app.PeakPlotMarkers{k}(m).Visible = newVis;
                        end
                    end
                end
                if k <= length(app.PeakPlotTexts) && ~isempty(app.PeakPlotTexts{k})
                    for m = 1:length(app.PeakPlotTexts{k})
                        if isvalid(app.PeakPlotTexts{k}(m))
                            app.PeakPlotTexts{k}(m).Visible = newVis;
                        end
                    end
                end
            else
                hitPeer.Visible = newVis;
            end
        end
        
        function updateSummaryStatus(app, chIndices)
            % Format dominant frequencies into status bar
            summaryTokens = {};
            for k = 1:min(length(chIndices), 3)
                col = chIndices(k);
                if ~isempty(app.DetectedPeaks{k})
                    pks = app.DetectedPeaks{k};
                    pkStrings = cellfun(@(f) sprintf('%.2fHz', f), {pks.frequency}, 'UniformOutput', false);
                    summaryTokens{end+1} = sprintf('%s: [%s]', app.ChannelNames{col}, strjoin(pkStrings, ', ')); %#ok<AGROW>
                end
            end
            
            if ~isempty(summaryTokens)
                app.StatusLabel.Text = sprintf('Identified Resonance Peaks: %s', strjoin(summaryTokens, ' | '));
            else
                app.StatusLabel.Text = 'Analysis updated. No prominent peaks detected with current thresholds.';
            end
        end
        
        % ---------------- Callbacks ----------------
        function onSelectFilePushed(app)
            % File chooser dialog
            filters = {'*.mat;*.csv;*.xlsx;*.xls;*.txt;*.dat', 'Supported Data Files (*.mat, *.csv, *.xlsx, *.txt)'; ...
                       '*.mat', 'MATLAB Files (*.mat)'; ...
                       '*.csv;*.txt;*.dat', 'Delimited Text Files (*.csv, *.txt)'; ...
                       '*.xlsx;*.xls', 'Excel Spreadsheets (*.xlsx, *.xls)'; ...
                       '*.*', 'All Files (*.*)'};
            [filename, pathname] = uigetfile(filters, 'Select Experimental Data File');
            if isequal(filename, 0)
                return;
            end
            fullPath = fullfile(pathname, filename);
            app.loadFile(fullPath);
        end
        
        function onTimeCheckChanged(app)
            if isempty(app.CurrentFilePath)
                return;
            end
            % Reload with explicit user override
            try
                isTime = app.FirstColTimeCheck.Value;
                [app.DataMatrix, app.ChannelNames, app.TimeVector, detectedFs, app.IsTimeDetected] = ...
                    SignalDataLoader.loadFile(app.CurrentFilePath, isTime);
                
                if isTime && ~isnan(detectedFs)
                    app.SamplingFreq = detectedFs;
                    app.FsEditField.Value = detectedFs;
                end
                
                if app.IsTimeDetected
                    app.FsEditField.Enable = 'off';
                    app.AutoFsButton.Enable = 'on';
                else
                    app.FsEditField.Enable = 'on';
                    app.AutoFsButton.Enable = 'off';
                end
                
                if isempty(app.TimeVector)
                    N = size(app.DataMatrix, 1);
                    app.TimeVector = (0:N-1)' / app.SamplingFreq;
                end
                
                [~, fname, fext] = fileparts(app.CurrentFilePath);
                app.FileLabel.Text = sprintf('%s%s (%d samples, %d channels)', ...
                    fname, fext, size(app.DataMatrix, 1), size(app.DataMatrix, 2));
                
                duration = size(app.DataMatrix, 1) / app.SamplingFreq;
                app.InfoLabel.Text = sprintf('Duration: %.2f s | Fs: %.1f Hz | Nyquist: %.1f Hz', ...
                                             duration, app.SamplingFreq, app.SamplingFreq/2);
                
                app.ChannelListBox.Items = app.ChannelNames;
                app.ChannelListBox.Value = app.ChannelNames;
                app.updateAnalysis();
            catch ME
                errordlg(ME.message, 'Reload Error');
            end
        end
        
        function onFsChanged(app)
            newFs = app.FsEditField.Value;
            if newFs > 0
                app.SamplingFreq = newFs;
                N = size(app.DataMatrix, 1);
                % Regenerate time vector if column 1 wasn't fixed time
                if ~app.FirstColTimeCheck.Value || isempty(app.TimeVector)
                    app.TimeVector = (0:N-1)' / newFs;
                end
                app.updateAnalysis();
            end
        end
        
        function onAutoFsPushed(app)
            if ~isempty(app.TimeVector) && length(app.TimeVector) > 1
                dt = median(diff(app.TimeVector));
                if dt > 0
                    autoFs = round(1.0 / dt, 4);
                    app.FsEditField.Value = autoFs;
                    app.SamplingFreq = autoFs;
                    app.updateAnalysis();
                    msgbox(sprintf('Estimated Fs = %.4f Hz (dt = %.6f s)', autoFs, dt), 'Sampling Rate Detected');
                end
            else
                warndlg('No time vector detected to calculate dt from.', 'Auto Fs');
            end
        end
        
        function onSelectAll(app)
            app.ChannelListBox.Value = app.ChannelListBox.Items;
            app.updateAnalysis();
        end
        
        function onClearAll(app)
            app.ChannelListBox.Value = {};
            app.updateAnalysis();
        end
        
        function onExportPlot(app)
            % Export publication-quality graphic
            [filename, pathname] = uiputfile({'*.png', 'Portable Network Graphics (*.png)'; ...
                                             '*.pdf', 'Vector PDF (*.pdf)'; ...
                                             '*.eps', 'Encapsulated PostScript (*.eps)'}, ...
                                             'Export Figure');
            if isequal(filename, 0), return; end
            outFile = fullfile(pathname, filename);
            
            selectedChannels = app.ChannelListBox.Value;
            if isempty(selectedChannels)
                warndlg('No channels selected to export.', 'Export Plot');
                return;
            end
            
            chIndices = [];
            for i = 1:length(selectedChannels)
                match = find(strcmp(app.ChannelNames, selectedChannels{i}));
                if ~isempty(match), chIndices(end+1) = match(1); end %#ok<AGROW>
            end
            if isempty(chIndices), return; end
            
            detrendVal = app.DetrendDropDown.Value;
            switch detrendVal
                case 'DC Offset / Mean', dMethod = 'mean';
                case 'Linear Trend', dMethod = 'linear';
                otherwise, dMethod = 'none';
            end
            
            exportFig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 1100, 800], 'Color', 'w');
            
            % Subplot 1: Time domain
            s1 = subplot(2, 1, 1, 'Parent', exportFig);
            hold(s1, 'on');
            t = app.TimeVector;
            numSel = length(chIndices);
            numColors = size(app.ColorPalette, 1);
            
            for k = 1:numSel
                col = chIndices(k);
                y = SignalProcessor.detrendSignal(app.DataMatrix(:, col), dMethod);
                c = app.ColorPalette(mod(col-1, numColors) + 1, :);
                plot(s1, t, y, 'LineWidth', 1.2, 'Color', c, 'DisplayName', app.ChannelNames{col});
            end
            hold(s1, 'off');
            grid(s1, 'on'); box(s1, 'on');
            title(s1, app.TimeAxes.Title.String, 'FontWeight', 'bold');
            xlabel(s1, app.TimeAxes.XLabel.String, 'FontWeight', 'bold');
            ylabel(s1, app.TimeAxes.YLabel.String, 'FontWeight', 'bold');
            xlim(s1, app.TimeAxes.XLim);
            legend(s1, 'Location', 'northeast');
            
            % Subplot 2: Freq domain
            s2 = subplot(2, 1, 2, 'Parent', exportFig);
            hold(s2, 'on');
            scaleMode = app.ScaleDropDown.Value;
            
            for k = 1:numSel
                col = chIndices(k);
                f = app.ComputedFreqs{k};
                switch scaleMode
                    case 'Decibels (dB)', spec = app.ComputeddB{k};
                    case 'PSD (Power Spectrum)', spec = app.ComputedPSD{k};
                    otherwise, spec = app.ComputedAmps{k};
                end
                c = app.ColorPalette(mod(col-1, numColors) + 1, :);
                plot(s2, f, spec, 'LineWidth', 1.4, 'Color', c, 'DisplayName', app.ChannelNames{col});
                
                if app.PeakCheck.Value && ~isempty(app.DetectedPeaks{k})
                    pks = app.DetectedPeaks{k};
                    for p = 1:length(pks)
                        pkFreq = pks(p).frequency;
                        if pkFreq <= app.FreqAxes.XLim(2)
                            [~, pIdx] = min(abs(f - pkFreq));
                            plot(s2, pkFreq, spec(pIdx), 'v', 'MarkerSize', 7, ...
                                 'MarkerEdgeColor', 'k', 'MarkerFaceColor', c, 'HandleVisibility', 'off');
                            text(s2, pkFreq, spec(pIdx), sprintf(' %.2f Hz', pkFreq), ...
                                 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.1], ...
                                 'VerticalAlignment', 'bottom');
                        end
                    end
                end
            end
            hold(s2, 'off');
            grid(s2, 'on'); box(s2, 'on');
            title(s2, app.FreqAxes.Title.String, 'FontWeight', 'bold');
            xlabel(s2, app.FreqAxes.XLabel.String, 'FontWeight', 'bold');
            ylabel(s2, app.FreqAxes.YLabel.String, 'FontWeight', 'bold');
            xlim(s2, app.FreqAxes.XLim);
            legend(s2, 'Location', 'northeast');
            
            try
                exportgraphics(exportFig, outFile, 'Resolution', 300);
                app.StatusLabel.Text = sprintf('Exported publication figure to %s', filename);
            catch
                saveas(exportFig, outFile);
                app.StatusLabel.Text = sprintf('Saved figure to %s', filename);
            end
            close(exportFig);
        end
        
        function onExportCSV(app)
            % Export spectral curves to CSV
            if isempty(app.ComputedFreqs)
                warndlg('No spectral data available to export.', 'Export');
                return;
            end
            [filename, pathname] = uiputfile('*.csv', 'Export Spectral Data to CSV');
            if isequal(filename, 0), return; end
            outFile = fullfile(pathname, filename);
            
            selectedChannels = app.ChannelListBox.Value;
            freq = app.ComputedFreqs{1};
            exportTable = table(freq, 'VariableNames', {'Frequency_Hz'});
            
            for k = 1:length(selectedChannels)
                safeName = matlab.lang.makeValidName(selectedChannels{k});
                exportTable.([safeName, '_LinearAmp']) = app.ComputedAmps{k};
                exportTable.([safeName, '_dB'])        = app.ComputeddB{k};
                exportTable.([safeName, '_PSD'])       = app.ComputedPSD{k};
            end
            
            writetable(exportTable, outFile);
            app.StatusLabel.Text = sprintf('Exported spectral data to %s', filename);
        end
    end
end

