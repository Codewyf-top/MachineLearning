import DrawDecisionTree.*
import DataHandling.*
import DesicionTreeLearning.*

normalise = false;
[cFtrs, cLbls, rFtrs, rLbls] = DataHandling(false);
cLblNames = ["No_Obesity", "Obesity"];

%% Classification Part
% 10-fold cross validation for classification
trAvgAcc = 0;
vaAvgAcc = 0;
trPrecAvg = [0,0];
trRecallAvg = [0,0];
trF1Avg = [0,0];
tePrecAvg = [0,0];
teRecallAvg = [0,0];
teF1Avg = [0,0];
k=10;
fprintf(1, strcat("\n", num2str(k), "-fold validation for classification \n"));
isClassification = true;
for i=1:k
    [trainFtrs, trainLbls, validFtrs, validLbls] = getKFoldData(cFtrs, cLbls, i, k);
    cTree = DecisionTreeLearning(trainFtrs, trainLbls, isClassification);
    fprintf(1, strcat("Fold-", num2str(i), "\n"));
    
    trAcc = accuracy(cTree, trainFtrs, trainLbls);
    trAvgAcc = trAvgAcc + trAcc;
    fprintf(1, strcat("Training Accuarcy:", num2str(trAcc), "\n"));
    
    [uniLbls, prec, recall, f1] = score(cTree ,trainFtrs, trainLbls);
    for j=1:length(uniLbls)
        fprintf(1, strcat("[Training set]", cLblNames(uniLbls(j)+1),...
                          " Precision: ", num2str(prec(j)), " ",...
                          "Recall: ", num2str(recall(j)), " ",...
                          "F1-score: ", num2str(f1(j)), "\n"));
        trPrecAvg(uniLbls(j)+1) = trPrecAvg(uniLbls(j)+1) + prec(j);
        trRecallAvg(uniLbls(j)+1) = trRecallAvg(uniLbls(j)+1) + recall(j);
        trF1Avg(uniLbls(j)+1) = trF1Avg(uniLbls(j)+1) + f1(j);
    end
   
    vaAcc = accuracy(cTree, validFtrs, validLbls);
    vaAvgAcc = vaAvgAcc + vaAcc;
    fprintf(1, strcat("Validating Accuarcy:", num2str(vaAcc), "\n"));
    
    [uniLbls, prec, recall, f1] = score(cTree ,validFtrs, validLbls);
    for j=1:length(uniLbls)
        fprintf(1, strcat("[Validating set]", cLblNames(uniLbls(j)+1),...
                      " Precision: ", num2str(prec(j)), " ",...
                      "Recall: ", num2str(recall(j)), " ",...
                      "F1-score: ", num2str(f1(j)), "\n"));
        tePrecAvg(uniLbls(j)+1) = tePrecAvg(uniLbls(j)+1) + prec(j);
        teRecallAvg(uniLbls(j)+1) = teRecallAvg(uniLbls(j)+1) + recall(j);
        teF1Avg(uniLbls(j)+1) = teF1Avg(uniLbls(j)+1) + f1(j);
    end
end
fprintf(1, strcat("\n", num2str(k), "-Fold Average", "\n"));
trAvgAcc = trAvgAcc / 10;
vaAvgAcc = vaAvgAcc / 10;
fprintf(1, strcat("Average training Accuarcy:", num2str(trAvgAcc), "\n"));
fprintf(1, strcat("Average validating Accuarcy:", num2str(vaAvgAcc), "\n"));
for i=1:length(cLblNames)
    [trPrecAvg(i), trRecallAvg(i), trF1Avg(i)] = deal((trPrecAvg(i) / k), (trRecallAvg(i) / k), (trF1Avg(i) / k));
    [tePrecAvg(i), teRecallAvg(i), teF1Avg(i)] = deal((tePrecAvg(i) / k), (teRecallAvg(i) / k), (teF1Avg(i) / k));
    fprintf(1, strcat("[Training]", cLblNames(uniLbls(i)+1),...
                    " Avg Precision: ", num2str(trPrecAvg(i)), " ",...
                    "Avg Recall: ", num2str(trRecallAvg(i)), " ",...
                    "Avg F1-score: ", num2str(trF1Avg(i)), "\n"));
    fprintf(1, strcat("[Validating]", cLblNames(uniLbls(i)+1),...
                " Avg Precision: ", num2str(tePrecAvg(i)), " ",...
                "Avg Recall: ", num2str(teRecallAvg(i)), " ",...
                "Avg F1-score: ", num2str(teF1Avg(i)), "\n"));
end

%% Regression Part
% 10-fold cross validation for regression
k=10;
mtRMSE = 0;
mvRMSE = 0;
fprintf(1, strcat(num2str(k), "-fold validation for regression \n"));
isClassification = false;
for i=1:k
    [trainFtrs, trainLbls, validFtrs, validLbls] = getKFoldData(rFtrs, rLbls, i, k);
    rTree = DecisionTreeLearning(trainFtrs, trainLbls, isClassification);
    trainRmse = RMSE(predict(rTree, trainFtrs), trainLbls);
    validRmse = RMSE(predict(rTree, validFtrs), validLbls);
    fprintf(1, strcat("Fold-", num2str(i), "\n"));
    fprintf(1, strcat(" RMSE of training set: ", num2str(trainRmse), " ", ...
                "RMSE of validating set: ", num2str(validRmse), "\n"));
    mtRMSE = mtRMSE + trainRmse;
    mvRMSE = mvRMSE + validRmse;
end
mtRMSE = mtRMSE / k;
mvRMSE = mvRMSE / k;
fprintf(1, strcat("\n", num2str(k), "-Fold Average", num2str(i), "\n"));
fprintf(1, strcat("Avg training RMSE: ", num2str(mtRMSE), "\n", ...
                "Avg validating RMSE: ", num2str(mvRMSE), "\n\n"));
         
%% F1-Score, RMSE and K-fold data split functions
% should be done with each class in labels
function [uniLbls, prec, recall, f1] = score(tree ,ftrs, lbls)
    uniLbls = unique(lbls);
    preds = predict(tree, ftrs); 
    prec = zeros(1,length(uniLbls));
    recall = zeros(1,length(uniLbls));
    f1 = zeros(1,length(uniLbls));
    for i=1:length(uniLbls)
        % ture_positives ture_negetives false_positives false_negetives 
        [tp, fp, tn, fn] = deal(0, 0, 0, 0);
        for j=1:length(preds)
            pred = preds(j, 1);
            real = lbls(j, 1);
            if pred == real
                if pred == uniLbls(i)
                    tp = tp + 1;
                else
                    tn = tn + 1;
                end
            else
                if pred == uniLbls(i)
                    fp = fp + 1;
                else
                    fn = fn + 1;
                end
            end
        end
        prec(i) = tp / (tp + fp);
        recall(i) = tp / (tp + fn);
        f1(i) = (2 * prec(i) * recall(i)) / (prec(i) + recall(i));
    end
end

function acc = accuracy(tree, ftrs, lbls)
    preds = predict(tree, ftrs); 
    acc = size(find(preds==lbls),1) / size(lbls,1);
end

function rmse = RMSE(preds, lbls)
    rmse = sqrt(mean((preds-lbls).^2));
end

function predictions = predict(root, ftrs)
    predictions = ones(size(ftrs, 1), 1);
    for i=1:size(ftrs, 1)
        tree = root;
        sample = ftrs(i,:);
        while ~isempty(tree.kids)
            if sample(tree.attribute) <= tree.threshold
                tree = tree.kids{1};
            else
                tree = tree.kids{2};
            end
        end
        predictions(i,1) = tree.prediction;
    end
end

function [trainFtrs, trainLbls, validFtrs, validLbls] = getKFoldData(ftrs, lbls, i, k)
    foldSize = idivide(length(ftrs), int32(k));
    trainFtrs = zeros((k-1)*foldSize, size(ftrs, 2));
    trainLbls = zeros((k-1)*foldSize, size(lbls, 2));
    stackCnt = 0;
    for j=1:k
        if j == i
            validFtrs = ftrs((j-1)*foldSize+1:j*foldSize, :); 
            validLbls = lbls((j-1)*foldSize+1:j*foldSize, :); 
        else
            trainFtrs(stackCnt*foldSize+1:(stackCnt+1)*foldSize, :) = ftrs((j-1)*foldSize+1:j*foldSize, :); 
            trainLbls(stackCnt*foldSize+1:(stackCnt+1)*foldSize, :) = lbls((j-1)*foldSize+1:j*foldSize, :); 
            stackCnt = stackCnt + 1;
        end
    end
end

