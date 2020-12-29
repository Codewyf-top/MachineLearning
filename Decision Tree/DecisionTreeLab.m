import DrawDecisionTree.*
import DataHandling.*
import DesicionTreeLearning.*
import RegressionTreeTrainingTest.*

[cFtrs, cLbls, rFtrs, rLbls] = DataHandling();

% Classification Part
fprintf(1, "First training of classification tree\n");
trainFtrs = cFtrs(1:1500, :);
trainLbls = cLbls(1:1500, :);
testFtrs = cFtrs(1501:end, :);
testLbls = cLbls(1501:end, :);
isClassification = true;
cTree = DecisionTreeLearning(trainFtrs, trainLbls, isClassification);
DrawDecisionTree(cTree, "Classification Tree");
[prec, recall, f1] = score(cTree ,trainFtrs, trainLbls);
fprintf(1, strcat("[Training set] Precision: ", num2str(prec), " ",...
                  "Recall: ", num2str(recall), " ",...
                  "F1-score: ", num2str(f1), "\n"));
[prec, recall, f1] = score(cTree ,testFtrs, testLbls);
fprintf(1, strcat("[Test set] Precision: ", num2str(prec), " ",...
                  "Recall: ", num2str(recall), " ",...
                  "F1-score: ", num2str(f1), "\n\n"));
              
% 10-fold cross validation for classification
[mtrPrec, mtrRecall, mtrF1] = deal(0, 0, 0);
[mtePrec, mteRecall, mteF1] = deal(0, 0, 0);
k=10;
fprintf(1, strcat(num2str(k), "-fold validation for classification \n"));
for i=1:k
[trainFtrs, trainLbls, validFtrs, validLbls] = getKFoldData(cFtrs, cLbls, i, k);
cTree = DecisionTreeLearning(trainFtrs, trainLbls, isClassification);
DrawDecisionTree(cTree, strcat("Fold-", num2str(i)," Classification Tree"));
[prec, recall, f1] = score(cTree ,trainFtrs, trainLbls);
fprintf(1, strcat("Fold-", num2str(i), "\n"));
fprintf(1, strcat("[Training] Precision: ", num2str(prec), " ",...
                  "Recall: ", num2str(recall), " ",...
                  "F1-score: ", num2str(f1), "\n"));
mtrPrec = mtrPrec + prec;
mtrRecall = mtrRecall + recall;
mtrF1 = mtrF1 + f1;
[prec, recall, f1] = score(cTree ,validFtrs, validLbls);
fprintf(1, strcat("[Test] Precision: ", num2str(prec), " ",...
                  "Recall: ", num2str(recall), " ",...
                  "F1-score: ", num2str(f1), "\n"));
mtePrec = mtePrec + prec;
mteRecall = mteRecall + recall;
mteF1 = mteF1 + f1;
end
[mtrPrec, mtrRecall, mtrF1] = deal((mtrPrec / k), (mtrRecall / k), (mtrF1 / k));
[mtePrec, mteRecall, mteF1] = deal((mtePrec / k), (mteRecall / k), (mteF1 / k));
fprintf(1, strcat("\n", num2str(k), "-Fold Average", "\n"));
fprintf(1, strcat("[Training] Avg Precision: ", num2str(mtrPrec), " ",...
                "Avg Recall: ", num2str(mtrRecall), " ",...
                "Avg F1-score: ", num2str(mtrF1), "\n"));
fprintf(1, strcat("[Test] Avg Precision: ", num2str(mtePrec), " ",...
                "Avg Recall: ", num2str(mteRecall), " ",...
                "Avg F1-score: ", num2str(mteF1), "\n\n"));

%% Regression Part
trainFtrs = rFtrs(1:1000, :);
trainLbls = rLbls(1:1000, :);
testFtrs = rFtrs(1001:end, :);
testLbls = rLbls(1001:end, :);
fprintf(1, "First training of regression tree\n");
isClassification = false;
rTree = DecisionTreeLearning(trainFtrs, trainLbls, isClassification);
DrawDecisionTree(rTree, "Regression Tree");
trainRmse = RMSE(predict(rTree, trainFtrs), trainLbls);
testRmse = RMSE(predict(rTree, testFtrs), testLbls);
fprintf(1, strcat(" RMSE of training set: ", num2str(trainRmse),...
             " RMSE of test set: ", num2str(testRmse), "\n\n"));
figure;
plot(floor(min(testLbls)):ceil(max(testLbls)), floor(min(testLbls)):ceil(max(testLbls)), 'b');hold on
plot(predict(rTree, testFtrs), testLbls, 'r*');
legend('Pefect Prediction Line');
xlabel('Predictions');
ylabel('Real Targets');
title('Predicted value and real value comparision');hold off

% 10-fold cross validation for regression
k=10;
mtRMSE = 0;
mvRMSE = 0;
fprintf(1, strcat(num2str(k), "-fold validation for regression \n"));
for i=1:k
    [trainFtrs, trainLbls, validFtrs, validLbls] = getKFoldData(rFtrs, rLbls, i, k);
    rTree = DecisionTreeLearning(trainFtrs, trainLbls, isClassification);
    DrawDecisionTree(rTree, strcat("Fold-", num2str(i)," Regression Tree"));
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
         
%% Hyperparameters testing for regression tree
trainFtrs = rFtrs(1:1000, :);
trainLbls = rLbls(1:1000, :);
testFtrs = rFtrs(1001:end, :);
testLbls = rLbls(1001:end, :);

% Do not draw the tree
doDraw = false;

% MaxDepth = 1:20
fprintf(1, "RMSE changing when Max Depth increasing\n");
trainRmseLog = zeros(20,1);
testRmseLog = zeros(20,1);
for i=1:20
    [maxDepth, minRss, minSamples] = deal(i,0,0);
    [trainRmseLog(i), testRmseLog(i)] = testRegressionParameters(trainFtrs, trainLbls,...
                                         testFtrs, testLbls, maxDepth, minRss, minSamples, doDraw);
end
figure;
plot(1:20, trainRmseLog, 'r');hold on
plot(1:20, testRmseLog, 'b');grid
legend('Training RMSE','Testing RMSE');
xlabel('Max Depth');
ylabel('RMSE');
title('RMSE changing when Max Depth increasing');hold off

% MinRSS = 1:20
trainRmseLog = [];
testRmseLog = [];
fprintf(1, "\nRMSE changing when MinRSS decreasing\n");
for i=20:-1:1
    [maxDepth, minRss, minSamples] = deal(16,i,0);
    [trainRmseLog(end+1), testRmseLog(end+1)] = testRegressionParameters(trainFtrs, trainLbls,...
                                             testFtrs, testLbls, maxDepth, minRss, minSamples, doDraw);
end
figure;
plot(20:-1:1, trainRmseLog, 'r');hold on
plot(20:-1:1, testRmseLog, 'b');grid
legend('Training RMSE','Testing RMSE');
xlabel('Minimal Residual Sum-of-Square');
ylabel('RMSE');
title('RMSE changing when MinRSS decreasing');hold off

% MinSamples = 1:20
trainRmseLog = zeros(20,1);
testRmseLog = zeros(20,1);
fprintf(1, "\nRMSE changing when MinSamples decreasing\n");
for i=20:-1:1
    [maxDepth, minRss, minSamples] = deal(16,0.001,i);
    [trainRmseLog(i), testRmseLog(i)] = testRegressionParameters(trainFtrs, trainLbls,...
                                        testFtrs, testLbls, maxDepth, minRss, minSamples, doDraw);
end
figure;
plot(20:-1:1, trainRmseLog, 'r');hold on
plot(20:-1:1, testRmseLog, 'b');grid
legend('Training RMSE','Testing RMSE');
xlabel('Minimal Sample number in leaf node');
ylabel('RMSE');
title('RMSE changing when MinSamples decreasing');hold off

% Test the combination of three parameters
fprintf(1, "\n Test the combination of three parameters\n");
isClassification = false;
[maxDepth, minRss, minSamples] = deal(11,9,9);
doDraw = true;
testRegressionParameters(trainFtrs, trainLbls, testFtrs, testLbls, maxDepth, minRss, minSamples, doDraw);

function [trainRmse, testRmse] = testRegressionParameters(trainFtrs, trainLbls, testFtrs, testLbls, maxDepth, minRss, minSamples, doDraw)
    rTree = RegressionTreeLearningTest(trainFtrs, trainLbls, maxDepth, minRss, minSamples);
    if doDraw
        DrawDecisionTree(rTree, "Classification Tree");
    end
    trainRmse = RMSE(predict(rTree, trainFtrs), trainLbls);
    testRmse = RMSE(predict(rTree, testFtrs), testLbls);
    fprintf(1, strcat("Max depth: ", num2str(maxDepth),...
                 " Min RSS of leaf node: ", num2str(minRss),...
                 " Min Samples of leaf node: ", num2str(minSamples),...
                 " RMSE of training set: ", num2str(trainRmse),...
                 " RMSE of test set: ", num2str(testRmse), "\n"));
    if doDraw
    figure;
    plot(floor(min(testLbls)):ceil(max(testLbls)), floor(min(testLbls)):ceil(max(testLbls)), 'b');hold on
    plot(predict(rTree, testFtrs), testLbls, 'r*');
    legend('Pefect Prediction Line');
    xlabel('Predictions');
    ylabel('Real Targets');
    title('Predicted value and real value comparision');hold off    
    end
end            

%% F1-Score, RMSE and K-fold data split functions
function [prec, recall, f1] = score(tree ,ftrs, lbls)
    preds = predict(tree, ftrs);
    % ture_positives ture_negetives false_positives false_negetives 
    [tp, fp, tn, fn] = deal(0, 0, 0, 0);
    for i=1:length(preds)
        pred = preds(i, 1);
        real = lbls(i, 1);
        if pred == real
            if pred == 1
                tp = tp + 1;
            else
                tn = tn + 1;
            end
        else
            if pred == 1
                fp = fp + 1;
            else
                fn = fn + 1;
            end
        end
    end
    prec = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = (2 * prec * recall) / (prec + recall);
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

