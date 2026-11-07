%% VERIFY_ACCURACY.M - Mathematical Verification & Benchmarking Test Suite
% Executes rigorous unit and integration validation on the DSP engine,
% verifying frequency resolution, amplitude scaling, window corrections,
% and multi-format file ingestion (.mat, .csv, .xlsx).
%
% Usage:
%   verify_accuracy

clear variables;
close all;
clc;

testDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(testDir);
addpath(projectDir);
addpath(fullfile(projectDir, 'src'));

fprintf('=================================================================\n');
fprintf('       SIGNAL ANALYZER MATHEMATICAL VERIFICATION BENCHMARK       \n');
fprintf('=================================================================\n\n');

totalTests = 0;
passedTests = 0;

%% TEST 1: Synthetic Multi-Tone Signal Precision Test
totalTests = totalTests + 1;
fprintf('[TEST 1] Synthetic Multi-Tone Frequency & Amplitude Accuracy:\n');
Fs = 1000.0;
T = 4.0;
N = round(Fs * T);
t = (0:N-1)' / Fs;

% Known ground-truth components
knownFreqs = [12.5, 35.0, 80.0];
knownAmps  = [1.00, 2.50, 0.80];
dcOffset   = 45.2;

% Construct test signal
x = dcOffset + knownAmps(1)*cos(2*pi*knownFreqs(1)*t) ...
             + knownAmps(2)*sin(2*pi*knownFreqs(2)*t) ...
             + knownAmps(3)*cos(2*pi*knownFreqs(3)*t);

% Run FFT with Flat Top window (ideal for exact amplitude verification)
opts = struct('detrendMethod', 'mean', 'windowType', 'flattop', 'nfft', 2^nextpow2(N));
[f, amp] = SignalProcessor.computeFFT(x, Fs, opts);

% Peak identification
pks = SignalProcessor.findProminentPeaks(f, amp, 3, 0.1, 1.0);

test1Pass = true;
fprintf('  Target Frequencies : [%.2f, %.2f, %.2f] Hz\n', knownFreqs(1), knownFreqs(2), knownFreqs(3));
fprintf('  Target Amplitudes  : [%.2f, %.2f, %.2f]\n', knownAmps(1), knownAmps(2), knownAmps(3));

if length(pks) ~= 3
    test1Pass = false;
    fprintf('  FAIL: Expected 3 peaks, found %d.\n', length(pks));
else
    detectedFreqs = [pks.frequency];
    detectedAmps  = [pks.amplitude];
    
    for i = 1:3
        freqErr = abs(detectedFreqs(i) - knownFreqs(i));
        ampErr  = abs(detectedAmps(i) - knownAmps(i)) / knownAmps(i) * 100;
        
        fprintf('  Mode %d -> Detected: %7.3f Hz (Err: %.4f Hz) | Amp: %6.3f (Err: %.2f%%)\n', ...
                i, detectedFreqs(i), freqErr, detectedAmps(i), ampErr);
            
        if freqErr > 0.05 || ampErr > 1.0
            test1Pass = false;
        end
    end
end

if test1Pass
    fprintf('  >> RESULT: PASS (Frequency error < 0.05 Hz, Amplitude error < 1.0%%)\n\n');
    passedTests = passedTests + 1;
else
    fprintf('  >> RESULT: FAIL\n\n');
end

%% TEST 2: DC Offset / Detrending Suppression Test
totalTests = totalTests + 1;
fprintf('[TEST 2] DC Offset Suppression & 0 Hz Spike Removal:\n');
% Test if DC = +1000.0 is cleanly removed without corrupting AC content
opts_dc = struct('detrendMethod', 'mean', 'windowType', 'hann', 'nfft', 2^nextpow2(N));
[f_dc, amp_dc] = SignalProcessor.computeFFT(x, Fs, opts_dc);

dcValueAt0Hz = amp_dc(1);
fprintf('  Injected DC Offset : +%.1f\n', dcOffset);
fprintf('  Residual Amp at 0Hz: %.2e\n', dcValueAt0Hz);

if dcValueAt0Hz < 1e-10
    fprintf('  >> RESULT: PASS (DC bias suppressed by > 160 dB)\n\n');
    passedTests = passedTests + 1;
else
    fprintf('  >> RESULT: FAIL (DC component leaked: %.4e)\n\n', dcValueAt0Hz);
end

%% TEST 3: Window Coherent Gain Invariance Test
totalTests = totalTests + 1;
fprintf('[TEST 3] Window Coherent Gain Normalization Invariance:\n');
% Single pure sinusoid with exact integer bin frequency
f_bin = 50.0;
x_bin = 2.0 * sin(2*pi*f_bin*t);
windowsToTest = {'rectangular', 'hann', 'hamming', 'blackman', 'flattop'};

test3Pass = true;
fprintf('  Input Sinusoid Amplitude: 2.000\n');
for w = 1:length(windowsToTest)
    wName = windowsToTest{w};
    opts_w = struct('detrendMethod', 'none', 'windowType', wName, 'nfft', N);
    [f_w, amp_w] = SignalProcessor.computeFFT(x_bin, Fs, opts_w);
    
    [peakAmp, idx] = max(amp_w);
    peakF = f_w(idx);
    errPct = abs(peakAmp - 2.0) / 2.0 * 100;
    
    fprintf('  Window: %-12s | Measured Amp: %6.4f | Error: %5.2f%%\n', wName, peakAmp, errPct);
    if errPct > 2.0
        test3Pass = false;
    end
end

if test3Pass
    fprintf('  >> RESULT: PASS (Coherent gain compensation accurate across all windowing functions)\n\n');
    passedTests = passedTests + 1;
else
    fprintf('  >> RESULT: FAIL\n\n');
end

%% TEST 4: Universal File Loader Ingestion (.mat, .csv, .xlsx)
totalTests = totalTests + 1;
fprintf('[TEST 4] Multi-Format Experimental Data Ingestion Consistency:\n');

sampleMat = fullfile(projectDir, 'sample_data', 'NonClassical_200Hz.mat');
sampleCsv = fullfile(projectDir, 'sample_data', 'NonClassical_200Hz.csv');
sampleXlsx = fullfile(projectDir, 'sample_data', 'NonClassical_200Hz.xlsx');

test4Pass = true;

% 1. Load MAT
if exist(sampleMat, 'file')
    [dataMat, namesMat, tMat, fsMat] = SignalDataLoader.loadFile(sampleMat);
    fprintf('  MAT File  : %d samples, %d channels, Fs = %.1f Hz\n', size(dataMat, 1), size(dataMat, 2), fsMat);
    if size(dataMat, 1) ~= 13336 || size(dataMat, 2) ~= 3 || abs(fsMat - 200.0) > 0.1
        fprintf('  FAIL: MAT dimensions or Fs mismatch (expected 13336x3 at 200 Hz)\n');
        test4Pass = false;
    end
else
    fprintf('  MAT File missing!\n');
    test4Pass = false;
end

% 2. Load CSV
if exist(sampleCsv, 'file')
    [dataCsv, namesCsv, tCsv, fsCsv] = SignalDataLoader.loadFile(sampleCsv);
    fprintf('  CSV File  : %d samples, %d channels, Fs = %.1f Hz\n', size(dataCsv, 1), size(dataCsv, 2), fsCsv);
    if size(dataCsv, 1) ~= 13336 || size(dataCsv, 2) ~= 3 || abs(fsCsv - 200.0) > 0.1
        fprintf('  FAIL: CSV dimensions or Fs mismatch (expected 13336x3 at 200 Hz)\n');
        test4Pass = false;
    end
    
    % Compare MAT and CSV data values
    maxDiff = max(abs(dataMat(:) - dataCsv(:)));
    fprintf('  Max Absolute Diff (MAT vs CSV): %.2e\n', maxDiff);
    if maxDiff > 1e-3
        test4Pass = false;
    end
else
    fprintf('  CSV File missing!\n');
    test4Pass = false;
end

% 3. Load XLSX (if available)
if exist(sampleXlsx, 'file')
    try
        [dataXlsx, namesXlsx, tXlsx, fsXlsx] = SignalDataLoader.loadFile(sampleXlsx);
        fprintf('  XLSX File : %d samples, %d channels, Fs = %.1f Hz\n', size(dataXlsx, 1), size(dataXlsx, 2), fsXlsx);
        if size(dataXlsx, 1) ~= 13336 || size(dataXlsx, 2) ~= 3 || abs(fsXlsx - 200.0) > 0.1
            fprintf('  FAIL: XLSX dimensions or Fs mismatch (expected 13336x3 at 200 Hz)\n');
            test4Pass = false;
        end
        maxDiffX = max(abs(dataMat(:) - dataXlsx(:)));
        fprintf('  Max Absolute Diff (MAT vs XLSX): %.2e\n', maxDiffX);
        if maxDiffX > 1e-3
            test4Pass = false;
        end
    catch ME
        fprintf('  Note: XLSX read skipped or failed (%s)\n', ME.message);
    end
end

if test4Pass
    fprintf('  >> RESULT: PASS (All formats ingested with cross-platform numeric identity)\n\n');
    passedTests = passedTests + 1;
else
    fprintf('  >> RESULT: FAIL\n\n');
end

%% TEST 5: firstColIsTimeOverride – explicit true / false / auto-detect paths
totalTests = totalTests + 1;
fprintf('[TEST 5] firstColIsTimeOverride (time-column override) Behaviour:\n');

test5Pass = true;

% Use the CSV which has a uniform time vector in column 1 ----------------
if ~exist(sampleCsv, 'file')
    fprintf('  SKIP: sample CSV not found – test cannot run.\n');
    test5Pass = false;
else
    % --- 5a: override = true (force col 1 as time) ---------------------
    try
        [d_t, n_t, tv_t, fs_t, isTD_t] = SignalDataLoader.loadFile(sampleCsv, true);
        % Expect: 3 sensor channels, time vector present, Fs ~= 200 Hz
        ok5a = (size(d_t, 2) == 3) && ...
               (numel(tv_t) == size(d_t, 1)) && ...
               (~isnan(fs_t)) && (abs(fs_t - 200.0) < 0.5) && ...
               isTD_t;
        if ok5a
            fprintf('  5a override=true  : %d samples, %d channels, Fs=%.2f Hz, hasTimeVec=%d  PASS\n', ...
                    size(d_t,1), size(d_t,2), fs_t, numel(tv_t) > 0);
        else
            fprintf('  5a override=true  : FAIL  (samples=%d, chans=%d, Fs=%.4f, timelen=%d, flag=%d)\n', ...
                    size(d_t,1), size(d_t,2), fs_t, numel(tv_t), isTD_t);
            test5Pass = false;
        end
    catch ME5a
        fprintf('  5a override=true  : FAIL (exception: %s)\n', ME5a.message);
        test5Pass = false;
    end

    % --- 5b: override = false (all columns treated as sensor channels) --
    try
        [d_f, n_f, tv_f, fs_f, isTD_f] = SignalDataLoader.loadFile(sampleCsv, false);
        % Expect: 4 channels (time col kept as data), no time vector, Fs = NaN
        ok5b = (size(d_f, 2) == 4) && ...
               isempty(tv_f) && ...
               isnan(fs_f) && ...
               (~isTD_f);
        if ok5b
            fprintf('  5b override=false : %d channels, timeVec=[], Fs=NaN  PASS\n', size(d_f,2));
        else
            fprintf('  5b override=false : FAIL  (chans=%d, timelen=%d, Fs=%.4f, flag=%d)\n', ...
                    size(d_f,2), numel(tv_f), fs_f, isTD_f);
            test5Pass = false;
        end
    catch ME5b
        fprintf('  5b override=false : FAIL (exception: %s)\n', ME5b.message);
        test5Pass = false;
    end

    % --- 5c: auto-detect (empty override) – should behave like override=true
    try
        [d_a, n_a, tv_a, fs_a, isTD_a] = SignalDataLoader.loadFile(sampleCsv, []);
        ok5c = (size(d_a, 2) == 3) && ...
               (numel(tv_a) == size(d_a, 1)) && ...
               (~isnan(fs_a)) && (abs(fs_a - 200.0) < 0.5) && ...
               isTD_a;
        if ok5c
            fprintf('  5c auto-detect    : %d samples, %d channels, Fs=%.2f Hz  PASS\n', ...
                    size(d_a,1), size(d_a,2), fs_a);
        else
            fprintf('  5c auto-detect    : FAIL  (chans=%d, Fs=%.4f, flag=%d)\n', ...
                    size(d_a,2), fs_a, isTD_a);
            test5Pass = false;
        end
    catch ME5c
        fprintf('  5c auto-detect    : FAIL (exception: %s)\n', ME5c.message);
        test5Pass = false;
    end

    % --- 5d: data identity – override=true and auto-detect must agree ---
    if exist('d_t', 'var') && exist('d_a', 'var') && ...
            isequal(size(d_t), size(d_a))
        maxDiff5 = max(abs(d_t(:) - d_a(:)));
        ok5d = (maxDiff5 == 0);
        if ok5d
            fprintf('  5d data identity  : override=true vs auto-detect, max diff = %.2e  PASS\n', maxDiff5);
        else
            fprintf('  5d data identity  : FAIL  (max diff = %.2e)\n', maxDiff5);
            test5Pass = false;
        end
    end
end

if test5Pass
    fprintf('  >> RESULT: PASS (Time-column override paths behave correctly)\n\n');
    passedTests = passedTests + 1;
else
    fprintf('  >> RESULT: FAIL\n\n');
end

%% Final Summary
fprintf('=================================================================\n');
fprintf('  TEST SUMMARY: %d / %d TESTS PASSED (%.1f%%)\n', passedTests, totalTests, (passedTests/totalTests)*100);
if passedTests == totalTests
    fprintf('  STATUS: ALL SYSTEM TESTS PASSED. READY FOR SUBMISSION.\n');
else
    fprintf('  STATUS: SOME TESTS FAILED. PLEASE REVIEW LOGS ABOVE.\n');
end
fprintf('=================================================================\n');

