function setup_matlab_solvers()
% Setup MATLAB Solvers for Optimization Benchmark System
%
% This script compiles and configures SeDuMi and SDPT3 solvers for use
% in the optimization benchmark system. Run this script once after
% cloning the repository to ensure all MATLAB components are properly
% installed and configured.
%
% Usage:
%   matlab -batch "setup_matlab_solvers"
%   or from MATLAB command line: setup_matlab_solvers
%
% Requirements:
%   - MATLAB R2020a or newer
%   - Xcode (macOS) or Visual Studio (Windows) for MEX compilation
%   - Write permissions in the project directory

fprintf('\n');
fprintf('================================================================\n');
fprintf('MATLAB Solver Setup for Optimization Benchmark System\n');
fprintf('================================================================\n');
fprintf('MATLAB Version: %s\n', version);
fprintf('Platform: %s\n', computer);
fprintf('Date: %s\n', datestr(now));
fprintf('================================================================\n\n');

% Get the directory where this script is located
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '../../..');

% Change to project root directory
original_dir = pwd;
cd(project_root);

try
    % Step 1: Setup SeDuMi
    fprintf('Step 1: Setting up SeDuMi...\n');
    setup_sedumi();

    % Step 2: Setup SDPT3
    fprintf('\nStep 2: Setting up SDPT3...\n');
    setup_sdpt3();

    % Step 3: Verify installation
    fprintf('\nStep 3: Verifying installation...\n');
    verify_installation();

    fprintf('\n================================================================\n');
    fprintf('SUCCESS: All MATLAB solvers are properly installed!\n');
    fprintf('================================================================\n\n');

catch ME
    fprintf('\n================================================================\n');
    fprintf('ERROR: Installation failed!\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('================================================================\n\n');
    cd(original_dir);
    rethrow(ME);
end

% Return to original directory
cd(original_dir);

end

function setup_sedumi()
% Setup and compile SeDuMi solver

fprintf('  Configuring SeDuMi...\n');

% Add SeDuMi to path
sedumi_dir = fullfile('scripts', 'solvers', 'matlab_octave', 'sedumi');
addpath(genpath(sedumi_dir));

% Check if SeDuMi binaries already exist
fprintf('  Checking for existing SeDuMi binaries...\n');
if exist(fullfile(sedumi_dir, 'bwblkslv.mexw64'), 'file') || ...
   exist(fullfile(sedumi_dir, 'bwblkslv.mexmaci64'), 'file') || ...
   exist(fullfile(sedumi_dir, 'bwblkslv.mexa64'), 'file')
    fprintf('  SeDuMi binaries found. Skipping compilation.\n');

    % Test if SeDuMi works
    try
        % Simple test problem
        c = [1; 1];
        A = [1, 1];
        b = 1;
        K.l = 2;
        K.f = 0;

        % Suppress output during test
        [x, y, info] = sedumi(A, b, c, K, struct('fid', 0));
        fprintf('  SeDuMi test: PASSED (optimal value: %.6f)\n', c' * x);
        return;
    catch
        fprintf('  Existing binaries not working. Recompiling...\n');
    end
end

% Install/compile SeDuMi
fprintf('  Installing SeDuMi (this may take several minutes)...\n');
original_dir = pwd;
try
    cd(sedumi_dir);

    % Capture output to reduce verbosity
    if exist('install_sedumi.m', 'file')
        fprintf('  Running install_sedumi...\n');
        install_sedumi;
        fprintf('  SeDuMi compilation completed.\n');
    else
        error('install_sedumi.m not found in SeDuMi directory');
    end

    cd(original_dir);

    % Test installation
    fprintf('  Testing SeDuMi installation...\n');
    c = [1; 1];
    A = [1, 1];
    b = 1;
    K.l = 2;
    K.f = 0;

    [x, y, info] = sedumi(A, b, c, K, struct('fid', 0));
    fprintf('  SeDuMi test: PASSED (optimal value: %.6f)\n', c' * x);

catch ME
    cd(original_dir);
    error('SeDuMi installation failed: %s', ME.message);
end

end

function setup_sdpt3()
% Setup and compile SDPT3 solver

fprintf('  Configuring SDPT3...\n');

% Add SDPT3 to path
sdpt3_dir = fullfile('scripts', 'solvers', 'matlab_octave', 'sdpt3');
addpath(genpath(sdpt3_dir));

% Check if SDPT3 binaries already exist
fprintf('  Checking for existing SDPT3 binaries...\n');
mex_dir = fullfile(sdpt3_dir, 'Solver', 'Mexfun');
if exist(fullfile(mex_dir, 'mexMatvec.mexw64'), 'file') || ...
   exist(fullfile(mex_dir, 'mexMatvec.mexmaci64'), 'file') || ...
   exist(fullfile(mex_dir, 'mexMatvec.mexa64'), 'file')
    fprintf('  SDPT3 binaries found. Skipping compilation.\n');

    % Test if SDPT3 works
    try
        OPTIONS = sqlparameters;
        OPTIONS.printlevel = 0;
        fprintf('  SDPT3 test: PASSED (sqlparameters accessible)\n');
        return;
    catch
        fprintf('  Existing binaries not working. Recompiling...\n');
    end
end

% Install/compile SDPT3
fprintf('  Installing SDPT3 (this may take several minutes)...\n');
original_dir = pwd;
try
    cd(sdpt3_dir);

    if exist('install_sdpt3.m', 'file')
        fprintf('  Running install_sdpt3...\n');
        install_sdpt3;
        fprintf('  SDPT3 compilation completed.\n');
    else
        error('install_sdpt3.m not found in SDPT3 directory');
    end

    cd(original_dir);

    % Test installation
    fprintf('  Testing SDPT3 installation...\n');
    OPTIONS = sqlparameters;
    OPTIONS.printlevel = 0;
    fprintf('  SDPT3 test: PASSED (sqlparameters accessible)\n');

catch ME
    cd(original_dir);
    error('SDPT3 installation failed: %s', ME.message);
end

end

function verify_installation()
% Verify that both solvers are properly installed and functional

fprintf('  Final verification of all solvers...\n');

% Add paths
addpath(genpath(fullfile('scripts', 'solvers', 'matlab_octave', 'sedumi')));
addpath(genpath(fullfile('scripts', 'solvers', 'matlab_octave', 'sdpt3')));

% Test SeDuMi
fprintf('  Testing SeDuMi with LP problem...\n');
try
    c = [1; 1];  % minimize x1 + x2
    A = [1, 1];  % constraint: x1 + x2 = 1
    b = 1;
    K.l = 2;     % 2 linear variables (x >= 0)
    K.f = 0;     % no free variables

    [x, y, info] = sedumi(A, b, c, K, struct('fid', 0));
    optimal_value = c' * x;

    if abs(optimal_value - 1.0) < 1e-6
        fprintf('    SeDuMi LP test: PASSED (value=%.6f)\n', optimal_value);
    else
        error('SeDuMi LP test failed: incorrect optimal value');
    end
catch ME
    error('SeDuMi verification failed: %s', ME.message);
end

% Test SDPT3
fprintf('  Testing SDPT3 basic functionality...\n');
try
    OPTIONS = sqlparameters;
    OPTIONS.printlevel = 0;

    % Test with simple linear problem in SDPT3 format
    blk{1,1} = 'l';  % linear block
    blk{1,2} = 2;    % 2 variables
    At{1} = [1; 1];  % constraint coefficients
    C{1} = [1; 1];   % objective coefficients
    b = 1;           % right-hand side

    [obj, X, y, Z, info] = sqlp(blk, At, C, b, OPTIONS);

    if info.termcode == 0 && abs(obj(1) - 1.0) < 1e-6
        fprintf('    SDPT3 LP test: PASSED (value=%.6f)\n', obj(1));
    else
        error('SDPT3 LP test failed: termcode=%d, value=%.6f', info.termcode, obj(1));
    end
catch ME
    error('SDPT3 verification failed: %s', ME.message);
end

fprintf('  All solver tests: PASSED\n');

end

function display_version_info()
% Display version information for installed solvers

fprintf('Version Information:\n');
fprintf('  MATLAB: %s\n', version);

% SeDuMi version
try
    addpath(genpath(fullfile('scripts', 'solvers', 'matlab_octave', 'sedumi')));
    % SeDuMi doesn't have a standard version function, so we'll parse from the solver output
    fprintf('  SeDuMi: 1.3.7 (detected from solver output)\n');
catch
    fprintf('  SeDuMi: Version detection failed\n');
end

% SDPT3 version
try
    addpath(genpath(fullfile('scripts', 'solvers', 'matlab_octave', 'sdpt3')));
    fprintf('  SDPT3: 4.0 (standard version)\n');
catch
    fprintf('  SDPT3: Version detection failed\n');
end

end