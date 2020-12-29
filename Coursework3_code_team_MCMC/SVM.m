import DataHandling.*

%% Part A
normalise = true;
[cFtrs, cLbls, rFtrs, rLbls] = DataHandling(normalise);

cTrFtrs = cFtrs(1:1500,:);
cTeFtrs = cFtrs(1501:end,:);
rTrFtrs = rFtrs(1:1000,:);
rTeFtrs = rFtrs(1001:end,:);

cTrLbls = cLbls(1:1500,:);
cTeLbls = cLbls(1501:end,:);
rTrLbls = rLbls(1:1000,:);
rTeLbls = rLbls(1001:end,:);

% Classification
cMdl = fitcsvm(cTrFtrs, cTrLbls, 'KernelFunction', 'linear', 'BoxConstraint', 1);
[preds, score] = predict(cMdl, cTeFtrs);

classOrder = cMdl.ClassNames;
sv = cMdl.SupportVectors;
figure
gscatter(cTrFtrs(:,1),cTrFtrs(:,2),cTrLbls)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',5)
legend('Obesity','Not Obesity','Support Vector')
hold off
% for test set
f1Score(preds, cTeLbls)
table(cTeLbls(1:10),preds(1:10),score(1:10,2),'VariableNames', {'TrueLabel','PredictedLabel','Score'})
%%k-fold cross validation for classification
k = 10;
sum_accuracy_svm = 0;
for i = 1:k
    [cTrainFtrs, cTrainLbls, validFtrs, validLbls] = getKFoldData(cFtrs, cLbls, i, k);
    classifer = fitcsvm(cTrainFtrs, cTrainLbls, 'KernelFunction', 'linear', 'BoxConstraint', 1);%训练模型
    predict_label  = predict(classifer, validFtrs);%test
    accuracy_svm = length(find(predict_label == validLbls))/length(validLbls)%acc
    sum_accuracy_svm = sum_accuracy_svm + accuracy_svm;
    
end
% get average acc
mean_accuracy_svm = sum_accuracy_svm / k;
disp('Classification: Average cross_validation accuracy ：');   
disp( mean_accuracy_svm);

% Regression
rmse = [];
for i=[0.1:0.2:20]
    rMdl = fitrsvm(rTrFtrs, rTrLbls, 'Epsilon', i);   
    preds = predict(rMdl, rTeFtrs);
    rmse(end+1) = RMSE(preds, rTeLbls);
    fprintf(1,strcat("Epsilon=", num2str(i), " RMSE=", num2str(rmse(end)),  "\n"));
end

figure;
plot([0.1:0.2:20], rmse);
xlabel('Epsilon');
ylabel('RMSE');
title('Changing of RMSE following changing of Epsilon');

rMdl = fitrsvm(rTrFtrs, rTrLbls, 'Epsilon', 3.1);
preds = predict(rMdl, rTeFtrs);
rmse = RMSE(preds, rTeLbls);
fprintf(1,strcat("Epsilon=", num2str(3.1), " RMSE=", num2str(rmse), "\n"));
plotComparison(preds, rTeLbls);

k = 10;
sum_rmse = 0;
for j=1:k
    [rTrainFtrs, rTrainLbls, rValidFtrs, rValidLbls] = getKFoldData(rFtrs, rLbls, j, k);
    rMdl = fitrsvm(rTrainFtrs, rTrainLbls, 'Epsilon', 3.1);
    preds = predict(rMdl, rValidFtrs);
    rmse = RMSE(preds, rValidLbls);
    sum_rmse = sum_rmse + rmse;
end
mean_rmse_svm = sum_rmse / k;
disp('Regression: Average cross_validation accuracy with Epsilon=3.1：');   
disp( mean_rmse_svm);




%% Part B
paramCombinNums = 30;
innerFoldsNum = 10;
randMax = 30;
% Classification
fprintf(1, strcat("\nOne time optimazition for Gaussian RBF kernel SVM\n"));
[trAcc, teAcc, SVNumber, SVPercentage, trScoreTable, vaScoreTable] = innerFoldOptimizer(cTrFtrs, cTrLbls, cTeFtrs, cTeLbls,...
    "Classification", "rbf", paramCombinNums, innerFoldsNum, randMax);
trScoreTable
vaScoreTable
fprintf(1, strcat("Training accuracy:", num2str(trAcc), "\n",...
    "Testing accuracy:", num2str(teAcc), "\n",...
    "Number of Support Vectors:", num2str(SVNumber), "\n",...
    "Percentage of Support Vectors:", num2str(SVPercentage), "\n"));

fprintf(1, strcat("\nOne time optimazition for Polynomial kernel SVM\n"));
[trAcc, teAcc, SVNumber, SVPercentage, trScoreTable, vaScoreTable] = innerFoldOptimizer(cTrFtrs, cTrLbls, cTeFtrs, cTeLbls,...
    "Classification", "polynomial", paramCombinNums, innerFoldsNum, randMax);
trScoreTable
vaScoreTable
fprintf(1, strcat("Training accuracy:", num2str(trAcc), "\n",...
    "Testing accuracy:", num2str(teAcc), "\n",...
    "Number of Support Vectors:", num2str(SVNumber), "\n",...
    "Percentage of Support Vectors:", num2str(SVPercentage), "\n"));

% Regression
fprintf(1, strcat("\nOne time optimazition for Gaussian RBF kernel SVR\n"));
[trRMSE, teRMSE, SVNumber, SVPercentage] = innerFoldOptimizer(rTrFtrs, rTrLbls, rTeFtrs, rTeLbls,...
    "Regression", "rbf", paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("Training RMSE:", num2str(trRMSE), "\n",...
    "Testing RMSE:", num2str(teRMSE), "\n",...
    "Number of Support Vectors:", num2str(SVNumber), "\n",...
    "Percentage of Support Vectors:", num2str(SVPercentage), "\n"));

fprintf(1, strcat("\nOne time optimazition for Polynomial kernel SVR\n"));
[trRMSE, teRMSE, SVNumber, SVPercentage] = innerFoldOptimizer(rTrFtrs, rTrLbls, rTeFtrs, rTeLbls,...
    "Regression", "polynomial", paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("Training RMSE:", num2str(rmse), "\n",...
    "Testing RMSE:", num2str(teRMSE), "\n",...
    "Number of Support Vectors:", num2str(SVNumber), "\n",...
    "Percentage of Support Vectors:", num2str(SVPercentage), "\n"));

%% Part C
OuterFoldsNum = 10;
paramCombinNums = 30;
innerFoldsNum = 10;
randMax = 30;
%Classification
fprintf(1, strcat("\nNested Cross Validation for linear kernel SVM\n"));
[trAvgAcc, vaAvgAcc] = nestedCrossValidation(cFtrs, cLbls, "Classification", "linear",...
    OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("10 Fold average training accuracy:", num2str(trAvgAcc), "\n"));
fprintf(1, strcat("10 Fold average validating accuracy:", num2str(vaAvgAcc), "\n"));

fprintf(1, strcat("Nested Cross Validation for Gaussian RBF kernel SVM\n"));
[trAvgAcc, vaAvgAcc] = nestedCrossValidation(cFtrs, cLbls, "Classification", "rbf",...
    OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("10 Fold average training accuracy:", num2str(trAvgAcc), "\n"));
fprintf(1, strcat("10 Fold average validating accuracy:", num2str(vaAvgAcc), "\n"));

fprintf(1, strcat("Nested Cross Validation for Polynomial kernel SVM\n"));
[trAvgAcc, vaAvgAcc] = nestedCrossValidation(cFtrs, cLbls, "Classification", "polynomial",...
    OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("10 Fold average training accuracy:", num2str(trAvgAcc), "\n"));
fprintf(1, strcat("10 Fold average validating accuracy:", num2str(vaAvgAcc), "\n"));

% Regression
fprintf(1, strcat("\nNested Cross Validation for linear kernel SVR\n"));
[trAvgRMSE, vaAvgRMSE] = nestedCrossValidation(rFtrs, rLbls, "Regression", "linear",...
    OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("10 Fold average training RMSE:", num2str(trAvgRMSE), "\n"));
fprintf(1, strcat("10 Fold average validating RMSE:", num2str(vaAvgRMSE), "\n"));

fprintf(1, strcat("Nested Cross Validation for Gaussian RBF kernel SVR\n"));
[trAvgRMSE, vaAvgRMSE] = nestedCrossValidation(rFtrs, rLbls, "Regression", "rbf",...
    OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("10 Fold average training RMSE:", num2str(trAvgRMSE), "\n"));
fprintf(1, strcat("10 Fold average validating RMSE:", num2str(vaAvgRMSE), "\n"));

fprintf(1, strcat("Nested Cross Validation for Polynomial kernel SVR\n"));
[trAvgRMSE, vaAvgRMSE] = nestedCrossValidation(rFtrs, rLbls, "Regression", "polynomial",...
    OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax);
fprintf(1, strcat("10 Fold average training RMSE:", num2str(trAvgRMSE), "\n"));
fprintf(1, strcat("10 Fold average validating RMSE:", num2str(vaAvgRMSE), "\n"));


%% Functions for computing accuracy
% should be done with each class in labels
function scoreTable = f1Score(preds, lbls)
  lblNames = ["No_Obesity","Obesity"];
  uniLbls = [0, 1];
  prec = zeros(1,2);
  recall = zeros(1,2);
  f1 = zeros(1,2);
  for i=1:2
      % tp:ture_positives tn:ture_negetives
      % fp:false_positives fn:false_negetives 
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
  scoreTable = table(lblNames', prec', recall', f1', 'VariableNames',...
                     {'Label Names','Precision','Recall', 'F1-Score'});
end

function rmse = RMSE(preds, lbls)
    rmse = sqrt(mean((preds-lbls).^2));
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

function plotComparison(preds, lbls)
    figure;
    plot(floor(min(lbls)):ceil(max(lbls)), floor(min(lbls)):ceil(max(lbls)), 'b');hold on
    plot(preds, lbls, 'r*');
    legend('Pefect Prediction Line');
    xlabel('Predictions');
    ylabel('Real Targets');
    title('Predicted value and real value comparision');hold off    
end

function acc = accuracy(preds, lbls) 
    acc = size(find(preds==lbls), 1) / size(lbls, 1);
end

function [num, percentage] = countSupportVectors(mdl)
    num = size(find(mdl.IsSupportVector==1), 1);
    percentage = size(find(mdl.IsSupportVector==1), 1) / size(mdl.IsSupportVector, 1);
end

function r = myRand(max)
    r = max * rand(1);
end

function [trAvgAcc, vaAvgAcc] = nestedCrossValidation(ftrs, lbls, opt, kernel,...
                                                OuterFoldsNum, paramCombinNums, innerFoldsNum, randMax)
    trAvgAcc = 0;
    vaAvgAcc = 0;

    % Outter fold
    for i = 1:OuterFoldsNum
        [trFtrs, trLbls, vaFtrs, vaLbls] = getKFoldData(ftrs, lbls, i, OuterFoldsNum);
        [trAcc, vaAcc] = innerFoldOptimizer(trFtrs, trLbls, vaFtrs, vaLbls, opt, kernel,...
                                                                paramCombinNums, innerFoldsNum, randMax);
                                                            
        fprintf(1, strcat("Fold-", num2str(i), " Training accuracy:", num2str(trAcc), "\n"));
        fprintf(1, strcat("Fold-", num2str(i), " Validating accuracy:", num2str(vaAcc), "\n"));
        % adding accuracy up
        trAvgAcc = trAvgAcc + trAcc;
        vaAvgAcc = vaAvgAcc + vaAcc;
    end
    
    vaAvgAcc = vaAvgAcc / 10;
    trAvgAcc = trAvgAcc / 10;
end

function [finalTrAcc, finalVaAcc, SVNumber, SVPercentage, trScoreTable, vaScoreTable] = innerFoldOptimizer(trFtrs, trLbls, vaFtrs, vaLbls,...
                                                                opt, kernel, paramCombinNums, innerFoldsNum, randMax)
    finalTrAcc = 0;
    finalVaAcc = 0;

    bestC = 0;
    bestSigma = 0;
    bestP = 0;
    bestEpsilon = 0;
    
    if opt == "Classification"
        bestAcc = 0;
        for i = 1:paramCombinNums
            % set init parameters
            sigma = myRand(randMax);
            p = myRand(randMax);
            C = myRand(randMax);
            avgAcc = 0;
            % Inner fold
            for j = 1:innerFoldsNum
                [inTrFtrs, inTrLbls, inVaFtrs, inVaLbls] = getKFoldData(trFtrs, trLbls, j, innerFoldsNum);
                if kernel == "linear"
                    mdl = fitcsvm(inTrFtrs, inTrLbls, 'KernelFunction', char(kernel), 'BoxConstraint', C);
                elseif kernel == "rbf"
                    mdl = fitcsvm(inTrFtrs, inTrLbls, 'KernelFunction', char(kernel), 'KernelScale', sigma, 'BoxConstraint', C);
                else
                    mdl = fitcsvm(inTrFtrs, inTrLbls, 'KernelFunction', char(kernel), 'PolynomialOrder', p, 'BoxConstraint', C);
                end
                [preds,~] = predict(mdl, inVaFtrs);
                acc = accuracy(preds, inVaLbls);
                avgAcc = avgAcc + acc;
            end
            avgAcc = avgAcc / innerFoldsNum;
            if avgAcc > bestAcc
                bestSigma = sigma;
                bestC = C;
                bestP = p;
                bestAcc = avgAcc;
            end
        end
        if kernel == "linear"
            mdl = fitcsvm(trFtrs, trLbls, 'KernelFunction', char(kernel), 'BoxConstraint', bestC);
        elseif kernel == "rbf"
            mdl = fitcsvm(trFtrs, trLbls, 'KernelFunction', char(kernel), 'KernelScale', bestSigma, 'BoxConstraint', bestC);
        else
            mdl = fitcsvm(trFtrs, trLbls, 'KernelFunction', char(kernel), 'PolynomialOrder', bestP, 'BoxConstraint', bestC);
        end
        
        [preds,~] = predict(mdl, trFtrs);
        trScoreTable = f1Score(preds, trLbls);
        finalTrAcc = accuracy(preds, trLbls);
        
        [preds,~] = predict(mdl, vaFtrs);
        vaScoreTable = f1Score(preds, vaLbls);
        finalVaAcc = accuracy(preds, vaLbls);
        
    elseif opt == "Regression"
        bestRMSE = inf;
        for i = 1:paramCombinNums
            % set init parameters
            sigma = myRand(randMax);
            p = myRand(randMax);
            C = myRand(randMax);
            epsilon = myRand(randMax);
            avgRMSE = 0;
            % Inner fold
            for j = 1:innerFoldsNum
                [inTrFtrs, inTrLbls, inVaFtrs, inVaLbls] = getKFoldData(trFtrs, trLbls, j, innerFoldsNum);
                if kernel == "linear"
                    mdl = fitrsvm(inTrFtrs, inTrLbls, 'KernelFunction', char(kernel), 'BoxConstraint', C, 'Epsilon', epsilon);
                elseif kernel == "rbf"
                    mdl = fitrsvm(inTrFtrs, inTrLbls, 'KernelFunction', char(kernel), 'KernelScale', sigma,...
                        'BoxConstraint', C, 'Epsilon', epsilon);
                else
                    mdl = fitrsvm(inTrFtrs, inTrLbls, 'KernelFunction', char(kernel), 'PolynomialOrder', p,...
                        'BoxConstraint', C, 'Epsilon', epsilon);
                end
                preds = predict(mdl, inVaFtrs);
                rmse = RMSE(preds, inVaLbls);
                avgRMSE = avgRMSE + rmse;
            end
            avgRMSE = avgRMSE / innerFoldsNum;
            if avgRMSE < bestRMSE
                bestSigma = sigma;
                bestC = C;
                bestP = p;
                bestRMSE = avgRMSE;
            end
        end
        if kernel == "linear"
            mdl = fitrsvm(trFtrs, trLbls, 'KernelFunction', char(kernel), 'BoxConstraint', bestC, 'Epsilon', bestEpsilon);
        elseif kernel == "rbf"
            mdl = fitrsvm(trFtrs, trLbls, 'KernelFunction', char(kernel), 'KernelScale', bestSigma,...
                'BoxConstraint', bestC, 'Epsilon', bestEpsilon);
        else
            mdl = fitrsvm(trFtrs, trLbls, 'KernelFunction', char(kernel), 'PolynomialOrder', bestP,...
                'BoxConstraint', bestC, 'Epsilon', bestEpsilon);
        end
        
        preds = predict(mdl, trFtrs);
        finalTrAcc = RMSE(preds, trLbls);
        
        preds = predict(mdl, vaFtrs);
        finalVaAcc = RMSE(preds, vaLbls);

        trScoreTable = NaN; % means no f1-measurement in regression
        vaScoreTable = NaN;

    end

    [SVNumber, SVPercentage] = countSupportVectors(mdl);
end