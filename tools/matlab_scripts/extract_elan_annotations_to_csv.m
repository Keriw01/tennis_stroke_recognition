clear; clc;

addpath('tools/matlab_scripts/ELAN m-funkcje');

eafFiles = {
  'data/annotations_elan/kTO7o5A0sys_30fps.eaf';
  'data/annotations_elan/ZVj-ukiIDJE_30fps.eaf';
  'data/annotations_elan/Tcly8G0MQDg_30fps.eaf';
  'data/annotations_elan/DGIkkdxMrWc_30fps.eaf';
  'data/annotations_elan/gRSRlMZbVYE_30fps.eaf'
};

frameRate = 30;
tierName = 'Hit_Player';

outputFolder = 'data/annotations_csv';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Initializing the summary table
combinedTable = table();

for i = 1:length(eafFiles)
    % Loading data from an EAF file
    oElan = LoadEAF(eafFiles{i}, frameRate);

    % Getting annotations from a layer
    oAnnotations = GetAnnotations(oElan, tierName);

    if isempty(oAnnotations)
        continue;
    end

    numAnn = length(oAnnotations);
    startFrames = zeros(numAnn, 1);
    endFrames = zeros(numAnn, 1);
    labels = strings(numAnn, 1);

    for j = 1:numAnn
        startFrames(j) = oAnnotations(j).iStartIndex;
        endFrames(j) = oAnnotations(j).iStopIndex;
        labels(j) = string(oAnnotations(j).sFields);
    end

    % Getting the file name without extension
    [~, fileName, ~] = fileparts(eafFiles{i});
    fileNames = repmat(string(fileName), numAnn, 1);

    T = table(fileNames, startFrames, endFrames, labels, ...
              'VariableNames', {'fileName', 'startFrame', 'endFrame', 'label'});

    csvFilePath = fullfile(outputFolder, [fileName '.csv']);
    writetable(T, csvFilePath);

    % Adding to the summary table
    combinedTable = [combinedTable; T];
end

% Saving a bulk CSV file
combinedCSVPath = fullfile(outputFolder, 'combined_annotations.csv');
writetable(combinedTable, combinedCSVPath);