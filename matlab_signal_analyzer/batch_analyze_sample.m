%% BATCH_ANALYZE_SAMPLE.M - Headless Analysis of Experimental Dataset
% Automates the end-to-end processing of the provided NonClassical_200Hz dataset,
% extracting resonant modal frequencies, damping ratios, and generating publication figures.
%
% Usage:
%   batch_analyze_sample

clear variables;
close all;
clc;

projectDir = fileparts(mfilename('fullpath'));
addpath(projectDir);
addpath(fullfile(projectDir, 'src'));

sampleMatPath = fullfile(projectDir, 'sample_data', 'NonClassical_200Hz.mat');
resultsDir    = fullfile(projectDir, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

fprintf('=================================================================\n');
fprintf('     AUTOMATED EXPERIMENTAL SIGNAL & MODAL SPECTRAL ANALYSIS     \n');
fprintf('=================================================================\n');
fprintf('Loading dataset: %s\n', sampleMatPath);

[dataMatrix, channelNames, timeVector, detectedFs, isTimeDetected] = ...
    SignalDataLoader.loadFile(sampleMatPath);

N = size(dataMatrix, 1);
numChannels = size(dataMatrix, 2);
Fs = detectedFs;
if isnan(Fs) || Fs <= 0
    Fs = 200.0; % Default for 200Hz dataset
end
if isempty(timeVector)
    timeVector = (0:N-1)' / Fs;
end
duration = N / Fs;

fprintf('Dataset successfully loaded.\n');
fprintf('  Total Samples (N) : %d\n', N);
fprintf('  Sampling Freq (Fs): %.2f Hz (dt = %.5f s)\n', Fs, 1/Fs);
fprintf('  Signal Duration   : %.2f seconds\n', duration);
fprintf('  Active Channels   : %d (%s)\n', numChannels, strjoin(channelNames, ', '));
fprintf('-----------------------------------------------------------------\n\n');

%% 1. Time-Domain Statistical Summary
fprintf('CHANNEL STATISTICAL CHARACTERISTICS:\n');
fprintf('%-15s %-12s %-12s %-12s %-12s\n', 'Channel', 'Mean (DC)', 'RMS', 'Peak-to-Peak', 'Std Dev');
fprintf('-----------------------------------------------------------------\n');
for c = 1:numChannels
    stats = SignalProcessor.computeSignalStats(dataMatrix(:, c), Fs);
    fprintf('%-15s %-12.3f %-12.3f %-12.3f %-12.3f\n', ...
            channelNames{c}, stats.Mean, stats.RMS, stats.PeakToPeak, stats.Std);
end
fprintf('-----------------------------------------------------------------\n\n');

%% 2. Spectral Analysis (FFT) with DC Detrending & Hann Window
dspOpts = struct();
dspOpts.detrendMethod = 'mean';
dspOpts.windowType    = 'hann';
dspOpts.nfft          = 2^nextpow2(N);
dspOpts.refAmp        = 1.0;

colorPalette = [
    0.0000, 0.4470, 0.7410;  % Blue
    0.8500, 0.3250, 0.0980;  % Orange
    0.9290, 0.6940, 0.1250;  % Yellow
    0.4940, 0.1840, 0.5560;  % Purple
];

freqResults = cell(1, numChannels);
ampResults  = cell(1, numChannels);
dbResults   = cell(1, numChannels);
psdResults  = cell(1, numChannels);
peaksTable  = {};

fprintf('IDENTIFIED DOMINANT RESONANCE PEAKS (0 - 100 Hz):\n');
fprintf('%-15s %-10s %-15s %-15s %-15s\n', 'Channel', 'Mode', 'Freq (Hz)', 'Amplitude', 'Damping (\zeta)');
fprintf('--------------------------------------------------------------------------------\n');

for c = 1:numChannels
    x = dataMatrix(:, c);
    [f, amp, psd, amp_dB] = SignalProcessor.computeFFT(x, Fs, dspOpts);
    
    freqResults{c} = f;
    ampResults{c}  = amp;
    dbResults{c}   = amp_dB;
    psdResults{c}  = psd;
    
    % Find top 4 resonant peaks above 0.1 Hz with at least 0.5 Hz separation
    pks = SignalProcessor.findProminentPeaks(f, amp, 4, 0.03, 0.1, 0.5);
    for p = 1:length(pks)
        fprintf('%-15s Mode %-5d %-15.3f %-15.4f %-15.4f\n', ...
                channelNames{c}, p, pks(p).frequency, pks(p).amplitude, pks(p).dampingRatio);
        peaksTable(end+1, :) = {channelNames{c}, p, pks(p).frequency, pks(p).amplitude, pks(p).dampingRatio}; %#ok<AGROW>
    end
end
fprintf('--------------------------------------------------------------------------------\n\n');

% Save peaks summary to CSV
if ~isempty(peaksTable)
    pTable = cell2table(peaksTable, 'VariableNames', {'Channel', 'ModeIndex', 'Frequency_Hz', 'Amplitude', 'DampingRatio'});
    writetable(pTable, fullfile(resultsDir, 'modal_peaks_summary.csv'));
    fprintf('Exported modal peak summary to: %s\n', fullfile(resultsDir, 'modal_peaks_summary.csv'));
end

%% 3. Plot 1: Time Domain Responses
fig1 = figure('Name', 'Time Domain Dynamic Response', 'Units', 'pixels', 'Position', [100, 100, 1000, 600], 'Visible', 'off');
hold on;
for c = 1:numChannels
    colColor = colorPalette(mod(c-1, size(colorPalette, 1)) + 1, :);
    plot(timeVector, dataMatrix(:, c), 'LineWidth', 1.2, 'Color', colColor, 'DisplayName', channelNames{c});
end
hold off;
grid on; box on;
title(sprintf('Multi-Channel Time-Domain Experimental Response (Fs = %.1f Hz)', Fs), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Sensor Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
xlim([timeVector(1), timeVector(end)]);
legend('Location', 'northeast');

timeFigPath = fullfile(resultsDir, 'time_domain_response.png');
exportgraphics(fig1, timeFigPath, 'Resolution', 300);
fprintf('Saved figure: %s\n', timeFigPath);
close(fig1);

%% 4. Plot 2: Frequency Domain Linear Amplitude Spectrum
fig2 = figure('Name', 'FFT Linear Amplitude Spectrum', 'Units', 'pixels', 'Position', [100, 100, 1000, 600], 'Visible', 'off');
hold on;
for c = 1:numChannels
    colColor = colorPalette(mod(c-1, size(colorPalette, 1)) + 1, :);
    plot(freqResults{c}, ampResults{c}, 'LineWidth', 1.3, 'Color', colColor, 'DisplayName', channelNames{c});
end
hold off;
grid on; box on;
title('Single-Sided FFT Linear Amplitude Spectrum (Detrended & Hann Windowed)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency (Hz)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Linear Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
xlim([0, Fs/2]);
legend('Location', 'northeast');

freqFigPath = fullfile(resultsDir, 'frequency_spectrum_linear.png');
exportgraphics(fig2, freqFigPath, 'Resolution', 300);
fprintf('Saved figure: %s\n', freqFigPath);
close(fig2);

%% 5. Plot 3: Frequency Domain Logarithmic Decibel (dB) Spectrum
fig3 = figure('Name', 'FFT Decibel Spectrum', 'Units', 'pixels', 'Position', [100, 100, 1000, 600], 'Visible', 'off');
hold on;
for c = 1:numChannels
    colColor = colorPalette(mod(c-1, size(colorPalette, 1)) + 1, :);
    plot(freqResults{c}, dbResults{c}, 'LineWidth', 1.3, 'Color', colColor, 'DisplayName', channelNames{c});
end
hold off;
grid on; box on;
title('Logarithmic Amplitude Spectrum (dB relative to 1.0)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency (Hz)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Magnitude (dB)', 'FontSize', 11, 'FontWeight', 'bold');
xlim([0, Fs/2]);
legend('Location', 'northeast');

dbFigPath = fullfile(resultsDir, 'frequency_spectrum_dB.png');
exportgraphics(fig3, dbFigPath, 'Resolution', 300);
fprintf('Saved figure: %s\n', dbFigPath);
close(fig3);

fprintf('\nBatch processing complete! All outputs generated in results/\n');

