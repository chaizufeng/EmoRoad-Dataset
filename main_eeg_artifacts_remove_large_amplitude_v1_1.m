%-------------------------------------------------------------------------%
% Description:                                                     
%   This code aims to remove components with large amplitude in EEG signals.
% Input:
%   csv file after cutting for time alignment
% Output:
%   Timestamp
%   OriginalTimestamp
%   32-chl eeg signals
% Author:
%   CHAI Bo & FANG Le (Leo)
%-------------------------------------------------------------------------%

clc; clear;

% settings
in_root  = 'path\to\Clip_EEG_noArtifact_icaCUS';   % input folder
out_root = 'path\to\Clip_EEG_noArtifact_icaCUS_thr100'; % output folder
amp_thresh = 100;   % 100μV threshold


% all participant
for part_NO = 1:50
    in_folder  = fullfile(in_root,  sprintf('P%d', part_NO));
    out_folder = fullfile(out_root, sprintf('P%d', part_NO));
    if ~exist(in_folder, 'dir')
        fprintf('Skip: %s (not found)\n', in_folder);
        continue;
    end
    if ~exist(out_folder, 'dir'); mkdir(out_folder); end

    files = dir(fullfile(in_folder, '*.csv'));
    for k = 1:numel(files)
        fin  = fullfile(in_folder, files(k).name);
        fout = fullfile(out_folder, files(k).name);

        % read csv, first 2 col are timestamp, rest are 32-chl
        T = readtable(fin, 'ReadVariableNames', true);
        if isempty(T)
            writetable(T, fout);
            fprintf('Empty file written: %s\n', fout);
            continue;
        end

        % pick out eeg data 
        eeg_mat = table2array(T(:, 3:end));  % N x 32
        if ~isfloat(eeg_mat); eeg_mat = double(eeg_mat); end

        % threshold mask
        mask_bad = any(abs(eeg_mat) > amp_thresh, 2);  % N x 1

        % remove bad data
        keep = ~mask_bad;
        T_clean = T(keep, :);

        % write to 
        writetable(T_clean, fout);
        fprintf('File written: %s | removed %d/%d (%.2f%%)\n', ...
            fout, sum(mask_bad), numel(mask_bad), 100*sum(mask_bad)/numel(mask_bad));
    end
end

