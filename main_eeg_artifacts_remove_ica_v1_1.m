%-------------------------------------------------------------------------%
% Description:                                                     
%   This code aims to use EEGLAB to remove the artifacts in EEG signals.
% Input:
%   csv file after cutting for time alignment
% Output:
%   Timestamp
%   OriginalTimestamp
%   32-chl eeg signals
% Author:
%   CHAI Bo & FANG Le (Leo)
%-------------------------------------------------------------------------%

clc;clear;
addpath('path\to\eeglab2025.1.0')

% open eeglab
eeglab;
chanlocs = pop_readlocs('path\to\Standard-10-20-Cap32.ced');  
channels = {chanlocs.labels};
header = ['Timestamp', 'OriginalTimestamp', channels];
for part_NO = 1:50
    folder_path = fullfile('path\to\Clip_EEG', sprintf('P%d', part_NO));  
    folder_path_out = fullfile('path\to\Clip_EEG_noArtifact_icaCUS', sprintf('P%d', part_NO));
    if ~exist(folder_path_out, 'dir')
        mkdir(folder_path_out);
    end
    files = dir(folder_path); 

    for i = 1:length(files)
        filename = files(i).name;

        % skip the '.' and '..'
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
            continue;
        end
        csv_data = readtable(fullfile(folder_path,filename),'ReadVariableNames', true);
        
        % check if empty
        if isempty(csv_data)

            T = cell2table(cell(0, numel(header)), 'VariableNames', header); % only write header

        else

            eeg_matrix = table2array(csv_data);
            eeg_data = eeg_matrix(:,5:36);  % extract 32-chl eeg data
            eeg_data = eeg_data';
                        
            % ica for artifact removal and filterign
            EEG = pop_importdata('dataformat', 'array', 'nbchan', 32, 'data', eeg_data, 'srate', 256, 'xmin', 0);
            EEG.chanlocs = chanlocs;
            EEG.dipfit.coordformat = 'CTF';
            total_duration = size(EEG.data, 2) / EEG.srate;

            % remove baseline
            EEG = pop_rmbase(EEG, [], []);
            
            % notch filter
            notchfreq = 50;
            bw = 2;  % notch bw
            EEG = pop_eegfiltnew(EEG, 'locutoff', notchfreq - bw/2, 'hicutoff', notchfreq + bw/2, 'revfilt', 1);

            % band pass filter
            EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5, 'hicutoff', 40);

            % ica
            EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'pca', 32); 
            
            % get IClabel (remove the artifacts whose probability is over threshold)
            EEG = iclabel(EEG);
            C = EEG.etc.ic_classification.ICLabel.classifications;
            thr = [NaN 0.6 0.6 0.7 0.7 0.7 NaN]; % [Brain Muscle Eye Line Channel Heart Other]
            
            nIC = size(C, 1);
            to_delete = false(1, nIC);
            
            for ic = 1:nIC  
                [pmax, cls] = max(C(ic, :));   % cls is index of class with highest possibility 
                t = thr(cls);                  % corresponding threshold
                if ~isnan(t) && pmax >= t
                    to_delete(ic) = true;
                end
            end
            
            artifact_components = find(to_delete);        
            EEG = pop_subcomp(EEG, artifact_components, 0);
    
            % prepare the data to be written into csv
            timestamp = eeg_matrix(:,1:2);  % copy timestamp & original timestamp
            data = [eeg_matrix(:,1:2), round(double(EEG.data'), 7)];
            T = array2table(data, 'VariableNames', header);
        end
        
        writetable(T,fullfile(folder_path_out,filename))
        fprintf('File written: %s\n', fullfile(folder_path_out, filename));
    end
end



