%% LAUNCH_APP.M - One-Click Launcher for Multi-Channel Signal Analyzer
% Run this script in MATLAB Command Window or Editor to start the application.
%
% Usage:
%   launch_app

clear variables;
close all;
clc;

appDir = fileparts(mfilename('fullpath'));
addpath(appDir);
addpath(fullfile(appDir, 'src'));

fprintf('-----------------------------------------------------------------\n');
fprintf('  Multi-Channel Experimental Signal & Spectral Analyzer (v1.0)   \n');
fprintf('  Time-Domain & Frequency-Domain Spectral Analysis Workstation   \n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Starting MATLAB App...\n');

app = SignalAnalyzerApp();

fprintf('Application window opened successfully.\n');

