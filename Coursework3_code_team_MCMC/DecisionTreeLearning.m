function tree = DecisionTreeLearning(ftrs, lbls, isClassification)
    if isClassification
        ftrNames = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF",...
            "TUE","Gender_Female","Gender_Male","family_history_with_overweight_no",...
            "family_history_with_overweight_yes","FAVC_no","FAVC_yes","CAEC_Always",...
            "CAEC_Frequently","CAEC_Sometimes","CAEC_no","SMOKE_no","SMOKE_yes",...
            "SCC_no","SCC_yes","CALC_Always","CALC_Frequently","CALC_Sometimes",...
            "CALC_no","MTRANS_Automobile","MTRANS_Bike","MTRANS_Motorbike",...
            "MTRANS_Public_Transportation","MTRANS_Walking"];
        lblNames = ["No_Obesity","Obesity"];
        [majorLbl, majorNum] = majorityValue(lbls);
        if majorNum == length(lbls)
            tree = cTree("", {}, majorLbl, nan, nan, majorNum, lblNames(majorLbl+1));
        else 
            [bestAtr, bestThld] = chooseAttribute(ftrs, lbls);
            [lData, rData] = splitData([ftrs lbls], bestAtr, bestThld);
            lSubtree = DecisionTreeLearning(lData(:, 1:end-1), lData(:, end), isClassification);
            rSubtree = DecisionTreeLearning(rData(:, 1:end-1), rData(:, end), isClassification);
            op = strcat(ftrNames{bestAtr}, "<=", string(bestThld));
            tree = cTree(op, {lSubtree rSubtree}, nan, bestAtr, bestThld, length(lbls), "");
        end 
    else
        ftrNames = ["Freq","Angle","Chord","FS_Velocity", "SSD_thickness"];
        stopCond = (length(unique(lbls))==1) || length(lbls)<= 9 || RSS(lbls)<= 9;
        if stopCond 
            tree = rTree("", {}, mean(lbls), nan, nan, length(lbls)); 
        else
            [bestAtr, bestThld] = rChooseAttribute(ftrs, lbls);
            [lData, rData] = splitData([ftrs lbls], bestAtr, bestThld);
            lSubtree = DecisionTreeLearning(lData(:, 1:end-1), lData(:, end), isClassification);
            rSubtree = DecisionTreeLearning(rData(:, 1:end-1), rData(:, end), isClassification);
            op = strcat(ftrNames{bestAtr}, "<=", string(bestThld));
            tree = rTree(op, {lSubtree rSubtree}, nan, bestAtr, bestThld, length(lbls));
        end
    end
end

function [majorLbl, majorNum] = majorityValue(lbls)
    uniqLbls = unique(lbls);
    majorLbl = 0;
    majorNum = 0;
    for i=1:size(uniqLbls, 1)
        nLbl = sum(lbls(:, 1) == uniqLbls(i));
        if majorNum < nLbl
            majorLbl = uniqLbls(i);
            majorNum = nLbl;
        end
    end
end

function [bestAtr, bestThld] = chooseAttribute(ftrs, lbls)
    tmpData = [ftrs lbls];
    maxGain = 0;
    baseEnt = entropy(lbls);
    for i=1:size(ftrs, 2)
        % column 9 to 31 of features are binary-class feature
        if i > 8
            thlds = 0.5;
        else 
            thlds = getThlds(ftrs(:, i));
        end
        for j=1:size(thlds, 1)
            [eltData, gtData] = splitData(tmpData, i, thlds(j, 1));
            newEnt = size(eltData, 1) / size(tmpData, 1) * entropy(eltData(:, end))...
                   + size(gtData, 1) / size(tmpData, 1) * entropy(gtData(:, end));
            gain = baseEnt - newEnt;
            if gain > maxGain
                maxGain = gain;
                bestAtr = i;
                bestThld = thlds(j, 1);
            end
        end
    end
end

function [bestAtr, bestThld] = rChooseAttribute(ftrs, lbls)
    tmpData = [ftrs lbls];
    minRss = inf;    
    for i=1:size(ftrs, 2)
        thlds = ftrs(:, i);
        for j=1:size(thlds, 1)
            [eltData, gtData] = splitData(tmpData, i, thlds(j, 1));
            if isempty(gtData)
                continue
            end
            rss = RSS(eltData(:, end)) + RSS(gtData(:, end));
            if rss < minRss
                minRss = rss;
                bestAtr = i;
                bestThld = thlds(j, 1);
            end
        end
    end
end

function ent = entropy(lbls)
    nRows = size(lbls, 1);
    uniqLbls = unique(lbls);
    ent = 0;
    for i=1:size(uniqLbls, 1)
        nLbl = sum(lbls(:, 1) == uniqLbls(i));
        prob = nLbl / nRows;
        ent = ent - prob * log2(prob);
    end
end

function rss = RSS(lbls)
    rss = sum((lbls-mean(lbls)).^2);
end

function [eltData, gtData] = splitData(data, atr, thld)
    eltData = data(data(:, atr)<=thld, :);
    gtData = data(data(:, atr)>thld, :);
end

function thlds = getThlds(ftr)
    thlds = zeros(size(ftr, 1)-1, 1);
    sortedFtr = sortrows(ftr);
    for i=1:size(ftr, 1)-1
        thlds(i, 1) = (sortedFtr(i) + sortedFtr(i+1)) / 2;
    end
end     

function tree = cTree(op, kids, pred, attr, thre, samp, predClass)
    tree.op = op;
    tree.kids = kids;
    tree.prediction = pred;
    tree.attribute =  attr;
    tree.threshold =  thre;
    tree.samples = samp;
    tree.predictionClass = predClass;
end

function tree = rTree(op, kids, pred, attr, thre, samp)
    tree.op = op;
    tree.kids = kids;
    tree.prediction = pred;
    tree.attribute =  attr;
    tree.threshold =  thre;
    tree.samples = samp;
end