function [cFtrs, cLbls, rFtrs, rLbls] = DataHandling()
    rng(1);

    % download classification dataset
    if ~exist('./ObesityDataSet_raw_and_data_sinthetic.csv', 'file')
        dataZipFile = websave('ObesityDataSet_raw_and_data_sinthetic.zip',...
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip');
        unzip(dataZipFile, '.');
        delete('./ObesityDataSet_raw_and_data_sinthetic.arff');
        delete('./ObesityDataSet_raw_and_data_sinthetic.zip');
    end
    
    % download regression dataset
    if ~exist('./airfoil_self_noise.dat', 'file')
    websave('airfoil_self_noise.dat',...
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat');
    rData = load('./airfoil_self_noise.dat');
    %randomise
    rData = rData(randperm(size(rData, 1)),:);
    writematrix(rData, './airfoil_self_noise.dat');
    end
    
    % Classification data pre-handling
    if ~exist('./ProcessedObesityDataset.csv', 'file')
        obesityTbl = readtable('./ObesityDataSet_raw_and_data_sinthetic.csv');
        % randomise
        obesityTbl = obesityTbl(randperm(size(obesityTbl, 1)),:);
        cFtrs = obesityTbl(:, 1:end-1);
        cFtrs = oneHotEncode(cFtrs);
        cLbls = obesityTbl(:, end); 
        cLbls = lblToBinary(cLbls);
        % lbls = lblToOrder(lbls);
        obesityTbl = [cFtrs cLbls];
        writetable(obesityTbl, 'ProcessedObesityDataset.csv');  
    else
        obesityTbl = readtable('./ProcessedObesityDataset.csv');
    end
    cFtrs = table2array(obesityTbl(:, 1:end-1));
    cLbls = table2array(obesityTbl(:, end)); 
    
    % Regression data pre-handling
    rData = load('./airfoil_self_noise.dat');
    rFtrs = rData(:, 1:end-1);
    rLbls = rData(:, end);
end

%% These are functions for data pre-handling

function newFtrs = oneHotEncode(ftrs)
    N = vartype('numeric');
    S = vartype('cellstr');
    newFtrs = ftrs(:, N);
    discFtrs = ftrs(:, S);
    for i=1:size(discFtrs, 2)
        ftrName = discFtrs.Properties.VariableNames{i};
        [classNames, oneHotKeyDict] = dictVectorizer(discFtrs(:, i));
        tmpMat = zeros(size(ftrs.(ftrName), 1), size(classNames, 2));
        for j=1:size(ftrs.(ftrName))
            class = ftrs.(ftrName){j};
            tmpMat(j,:) = oneHotKeyDict.(class);
        end
        tmpTbl = table(tmpMat, 'VariableNames', cellstr(ftrName));
        tmpTbl = splitvars(tmpTbl, ftrName, 'NewVariableNames',...
            cellfun(@cellstr, classNames));
        newFtrs = [newFtrs tmpTbl];
    end
end

function [classNames, dict] = dictVectorizer(ftr)
    ftrName = ftr.Properties.VariableNames;
    cntInfo = groupsummary(ftr, ftrName);
    classes = cntInfo(:, 1);
    classNames = cell(1, size(cntInfo, 1));
    mat = eye(size(classes, 1));
    for i=1:size(classes)
        className = string(classes{i,1});
        classNames{i} = strcat(ftrName, '_', className);
        dict.(className) = mat(i,:);
    end
end

function newLbls = lblToBinary(lbls)
    newLbls = table('Size', size(lbls), 'VariableTypes', "double",...
        'VariableNames', "NObeyesdad");
    for i = 1:height(lbls)
        switch lbls.NObeyesdad{i}
            case "Insufficient_Weight"
                newLbls.NObeyesdad(i) = 0;
            case "Normal_Weight"
                newLbls.NObeyesdad(i) = 0;
            case "Overweight_Level_I"
                newLbls.NObeyesdad(i) = 0;
            case "Overweight_Level_II"
                newLbls.NObeyesdad(i) = 0;
            case "Obesity_Type_I"
                newLbls.NObeyesdad(i) = 1;
            case "Obesity_Type_II"
                newLbls.NObeyesdad(i) = 1;
            case "Obesity_Type_III"
                newLbls.NObeyesdad(i) = 1;
        end
    end
end

function newLbls = lblToOrder(lbls)
    newLbls = table('Size', size(lbls), 'VariableTypes', "double",...
        'VariableNames', "NObeyesdad");
    for i = 1:height(lbls)
        switch lbls.NObeyesdad{i}
            case "Insufficient_Weight"
                newLbls.NObeyesdad(i) = 0;
            case "Normal_Weight"
                newLbls.NObeyesdad(i) = 1;
            case "Overweight_Level_I"
                newLbls.NObeyesdad(i) = 2;
            case "Overweight_Level_II"
                newLbls.NObeyesdad(i) = 3;
            case "Obesity_Type_I"
                newLbls.NObeyesdad(i) = 4;
            case "Obesity_Type_II"
                newLbls.NObeyesdad(i) = 5;
            case "Obesity_Type_III"
                newLbls.NObeyesdad(i) = 6;
        end
    end
end