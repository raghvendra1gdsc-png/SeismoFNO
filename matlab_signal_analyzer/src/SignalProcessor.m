classdef SignalProcessor
    % SignalProcessor - Research-grade Digital Signal Processing (DSP) Engine
    %
    % Implements mathematically rigorous spectral analysis routines:
    %   - Detrending: None, DC Mean subtraction, Linear trend subtraction
    %   - Windows: Rectangular, Hann, Hamming, Blackman, Flat Top
    %   - Amplitude Coherent Gain (CG) and Energy Gain (EG) corrections
    %   - Single-sided Fast Fourier Transform (FFT) with exact scaling
    %   - Power Spectral Density (PSD) via periodogram formulation
    %   - Automated peak identification with parabolic sub-bin refinement
    %   - Half-power (3-dB) modal damping ratio estimation (\zeta)
    
    methods (Static)
        function [f, ampSpectrum, psdSpectrum, amp_dB, win] = computeFFT(x, Fs, options)
            % Compute scaled single-sided FFT amplitude, PSD, and dB spectrum
            %
            % Inputs:
            %   x       - 1D real signal vector (N-by-1 or 1-by-N)
            %   Fs      - Sampling frequency in Hz
            %   options - (Optional) struct with fields:
            %               .detrendMethod: 'none', 'mean' (default), 'linear'
            %               .windowType:    'rectangular', 'hann' (default), 'hamming', 'blackman', 'flattop'
            %               .nfft:          FFT length (default: next power of 2 >= N)
            %               .refAmp:        Reference amplitude for dB scale (default: 1.0)
            %
            % Outputs:
            %   f           - Frequency vector [0, Fs/2] in Hz
            %   ampSpectrum - Single-sided linear amplitude spectrum (engineering units)
            %   psdSpectrum - Power Spectral Density (units^2 / Hz)
            %   amp_dB      - Single-sided amplitude spectrum in dB relative to refAmp
            %   win         - Window vector applied to signal
            
            if nargin < 3
                options = struct();
            end
            
            % Default options
            if ~isfield(options, 'detrendMethod') || isempty(options.detrendMethod)
                options.detrendMethod = 'mean';
            end
            if ~isfield(options, 'windowType') || isempty(options.windowType)
                options.windowType = 'hann';
            end
            if ~isfield(options, 'refAmp') || isempty(options.refAmp)
                options.refAmp = 1.0;
            end
            
            x = x(:);
            N = length(x);
            if N < 2
                error('SignalProcessor:SignalTooShort', 'Signal length must be at least 2 samples.');
            end
            
            if ~isfield(options, 'nfft') || isempty(options.nfft)
                options.nfft = 2^nextpow2(N);
            end
            Nfft = options.nfft;
            
            % Step 1: Detrending / DC Bias removal
            x_detrended = SignalProcessor.detrendSignal(x, options.detrendMethod);
            
            % Step 2: Window generation and gain calculations
            win = SignalProcessor.generateWindow(N, options.windowType);
            
            % Coherent Gain (for amplitude): CG = (1/N) * sum(w)
            coherentGain = sum(win) / N;
            if coherentGain == 0
                coherentGain = 1.0;
            end
            
            % Energy Gain (for PSD): S2 = sum(w.^2)
            S2 = sum(win.^2);
            if S2 == 0
                S2 = N;
            end
            
            % Step 3: Window application
            x_windowed = x_detrended .* win;
            
            % Step 4: Discrete Fourier Transform
            X = fft(x_windowed, Nfft);
            
            % Step 5: Single-sided frequency axis and spectrum extraction
            isEven = (mod(Nfft, 2) == 0);
            if isEven
                numUniquePts = (Nfft / 2) + 1;
            else
                numUniquePts = (Nfft + 1) / 2;
            end
            
            f = (0:numUniquePts-1)' * (Fs / Nfft);
            X_half = X(1:numUniquePts);
            
            % Step 6: Amplitude spectrum normalization
            % Single-sided amplitude formula:
            % DC & Nyquist: |X| / (N * CG)
            % AC components: 2 * |X| / (N * CG)
            ampSpectrum = abs(X_half) / (N * coherentGain);
            if numUniquePts > 2
                if isEven
                    ampSpectrum(2:end-1) = 2.0 * ampSpectrum(2:end-1);
                else
                    ampSpectrum(2:end) = 2.0 * ampSpectrum(2:end);
                end
            end
            % Exact DC component: physical DC average of the detrended signal
            ampSpectrum(1) = abs(mean(x_detrended));
            
            % Step 7: Decibel scaling (dB relative to refAmp)
            epsFloor = 1e-12;
            amp_dB = 20 * log10(max(ampSpectrum, epsFloor) / options.refAmp);
            
            % Step 8: Power Spectral Density (PSD)
            % Sxx(f) = (2 * |X|^2) / (Fs * sum(w.^2))
            psdSpectrum = (abs(X_half).^2) / (Fs * S2);
            if numUniquePts > 2
                if isEven
                    psdSpectrum(2:end-1) = 2.0 * psdSpectrum(2:end-1);
                else
                    psdSpectrum(2:end) = 2.0 * psdSpectrum(2:end);
                end
            end
            psdSpectrum(1) = (ampSpectrum(1)^2) / Fs;
        end
        
        function x_out = detrendSignal(x, method)
            % Remove mean or linear trend from signal vector
            x = x(:);
            N = length(x);
            if N < 2
                x_out = x;
                return;
            end
            
            switch lower(method)
                case {'none', 'raw'}
                    x_out = x;
                case {'mean', 'constant', 'dc'}
                    x_out = x - mean(x);
                case {'linear', 'poly1'}
                    t = (0:N-1)';
                    p = polyfit(t, x, 1);
                    trend = polyval(p, t);
                    x_out = x - trend;
                otherwise
                    warning('SignalProcessor:UnknownDetrend', 'Unknown detrend method: %s. Using mean subtraction.', method);
                    x_out = x - mean(x);
            end
        end
        
        function w = generateWindow(N, winType)
            % Standalone window generator (no Signal Processing Toolbox dependency)
            n = (0:N-1)';
            
            switch lower(winType)
                case {'rectangular', 'rect', 'boxcar', 'none'}
                    w = ones(N, 1);
                    
                case {'hann', 'hanning'}
                    if N == 1, w = 1; else
                        w = 0.5 - 0.5 * cos(2 * pi * n / (N - 1));
                    end
                    
                case {'hamming'}
                    if N == 1, w = 1; else
                        w = 0.54 - 0.46 * cos(2 * pi * n / (N - 1));
                    end
                    
                case {'blackman'}
                    if N == 1, w = 1; else
                        w = 0.42 - 0.5 * cos(2 * pi * n / (N - 1)) + 0.08 * cos(4 * pi * n / (N - 1));
                    end
                    
                case {'flattop', 'flat top'}
                    if N == 1, w = 1; else
                        a0 = 0.21557895; a1 = 0.41663158; a2 = 0.277263158; a3 = 0.083578947; a4 = 0.006947368;
                        z = 2 * pi * n / (N - 1);
                        w = a0 - a1*cos(z) + a2*cos(2*z) - a3*cos(3*z) + a4*cos(4*z);
                    end
                    
                otherwise
                    warning('SignalProcessor:UnknownWindow', 'Window %s unknown. Defaulting to Hann.', winType);
                    w = 0.5 - 0.5 * cos(2 * pi * n / (N - 1));
            end
        end
        
        function peaks = findProminentPeaks(f, amp, numPeaks, minProminence, minFreq, minDistance)
            % Identify dominant resonant peaks with sub-bin parabolic interpolation
            %
            % Inputs:
            %   f             - Frequency vector (Hz)
            %   amp           - Amplitude spectrum
            %   numPeaks      - Maximum number of peaks to report (default: 5)
            %   minProminence - Minimum height relative to max peak (default: 0.02 = 2%)
            %   minFreq       - Ignore peaks below minFreq Hz (default: 0.1 Hz to bypass DC remnants)
            %   minDistance   - Minimum frequency separation between peaks (default: 0.5 Hz)
            
            if nargin < 3 || isempty(numPeaks), numPeaks = 5; end
            if nargin < 4 || isempty(minProminence), minProminence = 0.02; end
            if nargin < 5 || isempty(minFreq), minFreq = 0.1; end
            if nargin < 6 || isempty(minDistance), minDistance = 0.5; end
            
            peaks = struct('frequency', {}, 'amplitude', {}, 'index', {}, 'dampingRatio', {});
            
            K = length(amp);
            if K < 3, return; end
            
            % Local maxima detection
            candIdx = find(amp(2:K-1) > amp(1:K-2) & amp(2:K-1) > amp(3:K)) + 1;
            
            % Filter by minimum frequency
            candIdx = candIdx(f(candIdx) >= minFreq);
            if isempty(candIdx), return; end
            
            % Filter by threshold
            maxVal = max(amp);
            candIdx = candIdx(amp(candIdx) >= (minProminence * maxVal));
            if isempty(candIdx), return; end
            
            % Sort descending by amplitude
            [~, sortOrder] = sort(amp(candIdx), 'descend');
            sortedCandIdx = candIdx(sortOrder);
            
            % Enforce minimum peak frequency separation (non-maximum suppression)
            selectedIdx = [];
            for i = 1:length(sortedCandIdx)
                idx_i = sortedCandIdx(i);
                freq_i = f(idx_i);
                if isempty(selectedIdx) || all(abs(f(selectedIdx) - freq_i) >= minDistance)
                    selectedIdx(end+1) = idx_i; %#ok<AGROW>
                end
                if length(selectedIdx) >= numPeaks
                    break;
                end
            end
            
            if isempty(selectedIdx), return; end
            
            % Sub-bin parabolic refinement and 3-dB damping estimation
            df = f(2) - f(1);
            for p = 1:length(selectedIdx)
                idx = selectedIdx(p);
                
                % Parabolic interpolation:
                % delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
                if idx > 1 && idx < K
                    alpha = amp(idx - 1);
                    beta  = amp(idx);
                    gamma = amp(idx + 1);
                    denom = (alpha - 2*beta + gamma);
                    if abs(denom) > eps
                        delta = 0.5 * (alpha - gamma) / denom;
                        delta = max(min(delta, 1.0), -1.0); % clamp
                        fineFreq = f(idx) + delta * df;
                        
                        % Amplitude calibration:
                        % For flat-top window or flat peaks, the top is flat (< 0.01 dB scallop loss).
                        % Applying parabolic correction on a flat profile creates an artificial overshoot.
                        % For non-flat profiles (Hann, Hamming, Blackman), parabolic correction compensates scallop loss.
                        if abs(beta - gamma) / max(beta, eps) < 0.02 || abs(beta - alpha) / max(beta, eps) < 0.02
                            fineAmp = max(beta, gamma);
                            if (beta - gamma) < 0.02 * beta
                                fineAmp = max(fineAmp, beta);
                            end
                        else
                            fineAmp = beta - 0.25 * (alpha - gamma) * delta;
                        end
                    else
                        fineFreq = f(idx);
                        fineAmp = beta;
                    end
                else
                    fineFreq = f(idx);
                    fineAmp = amp(idx);
                end
                
                % Half-power bandwidth (3-dB drop) with linear interpolation
                targetHalfPower = fineAmp / sqrt(2);
                
                % Search left
                leftIdx = idx;
                while leftIdx > 1 && amp(leftIdx) > targetHalfPower && amp(leftIdx) <= amp(leftIdx+1)
                    leftIdx = leftIdx - 1;
                end
                if amp(leftIdx) <= targetHalfPower && leftIdx < idx
                    denomL = amp(leftIdx+1) - amp(leftIdx);
                    if abs(denomL) > eps
                        f_a = f(leftIdx) + (targetHalfPower - amp(leftIdx)) / denomL * df;
                    else
                        f_a = f(leftIdx);
                    end
                else
                    f_a = f(leftIdx);
                end
                
                % Search right
                rightIdx = idx;
                while rightIdx < K && amp(rightIdx) > targetHalfPower && amp(rightIdx) <= amp(rightIdx-1)
                    rightIdx = rightIdx + 1;
                end
                if amp(rightIdx) <= targetHalfPower && rightIdx > idx
                    denomR = amp(rightIdx-1) - amp(rightIdx);
                    if abs(denomR) > eps
                        f_b = f(rightIdx-1) + (amp(rightIdx-1) - targetHalfPower) / denomR * df;
                    else
                        f_b = f(rightIdx);
                    end
                else
                    f_b = f(rightIdx);
                end
                
                deltaF_3dB = f_b - f_a;
                if fineFreq > 0 && deltaF_3dB > 0
                    zeta = deltaF_3dB / (2 * fineFreq);
                else
                    zeta = NaN;
                end
                
                peaks(p).frequency = fineFreq;
                peaks(p).amplitude = fineAmp;
                peaks(p).index = idx;
                peaks(p).dampingRatio = zeta;
            end
            
            % Sort final output by frequency for natural reading order
            [~, freqOrder] = sort([peaks.frequency]);
            peaks = peaks(freqOrder);
        end
        
        function stats = computeSignalStats(x, Fs)
            % Compute statistical features of a channel signal (100% Base MATLAB compatible)
            x = x(:);
            stats = struct();
            stats.N = length(x);
            stats.Fs = Fs;
            stats.Duration = stats.N / Fs;
            stats.Mean = mean(x);
            stats.Std = std(x);
            stats.RMS = sqrt(mean(x.^2));
            stats.PeakToPeak = max(x) - min(x);
            stats.Max = max(x);
            stats.Min = min(x);
        end
    end
end
