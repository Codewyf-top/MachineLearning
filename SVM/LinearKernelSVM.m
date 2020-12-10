import DataHandling.*

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

%% Classification
cMdl = fitcsvm(cTrFtrs, cTrLbls, 'KernelFunction', 'linear', 'BoxConstraint', 1);
[preds, score] = predict(cMdl, cTeFtrs);

% CVSVMModel = crossval(cMdl);
% classloss = kfoldLoss(CVSVMModel);
% str_c = sprintf('%0.5f',classloss);

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
[m,n] = size(cFtrs);
indices = crossvalind('Kfold',m,k);
for i = 1:k
    test_indic = (indices == i);
    train_indic = ~test_indic;
    train_datas = cFtrs(train_indic,:);
    train_labels = cLbls(train_indic,:);
    test_datas = cFtrs(test_indic,:);
    test_labels = cLbls(test_indic,:);
        
    classifer = fitcsvm(train_datas, train_labels, 'KernelFunction', 'linear', 'BoxConstraint', 1);%训练模型
    predict_label  = predict(classifer, test_datas);%test
    accuracy_svm = length(find(predict_label == test_labels))/length(test_labels)%acc
    sum_accuracy_svm = sum_accuracy_svm + accuracy_svm;
end
% get average acc
mean_accuracy_svm = sum_accuracy_svm / k;
disp('Average cross_validation accuracy：');   
disp( mean_accuracy_svm);
%% Regression
rmse = [];
for i=[0.1:0.2:20]
    rMdl = fitrsvm(rTrFtrs, rTrLbls, 'Epsilon', i);
    CVSVMModel = crossval(cMdl);
    
    classloss_r = kfoldLoss(CVSVMModel);

    
    preds = predict(rMdl, rTeFtrs);
    rmse(end+1) = RMSE(preds, rTeLbls);
    fprintf(1,strcat("Epsilon=", num2str(i), " RMSE=", num2str(rmse(end)), " kfold=", num2str(classloss_r), "\n"));
end
figure;
plot([0.1:0.2:20], rmse);
xlabel('Epsilon');
ylabel('RMSE');
title('Changing of RMSE following changing of Epsilon');

rMdl = fitrsvm(rTrFtrs, rTrLbls, 'Epsilon', 3.1);

CVSVMModel = crossval(cMdl);
classloss_r = kfoldLoss(CVSVMModel);

preds = predict(rMdl, rTeFtrs);
rmse = RMSE(preds, rTeLbls);
fprintf(1,strcat("Epsilon=", num2str(3.1), " RMSE=", num2str(rmse), " kfold=", num2str(classloss_r), "\n"));
plotComparison(preds, rTeLbls);


%% Functions for computing accuracy
% should be done with each class in labels
function scoreTable = f1Score(preds, lbls)
  lblNames = ["No_Obesity","Obesity"];
  uniLbls = [0, 1];
  prec = zeros(1,2);
  recall = zeros(1,2);
  f1 = zeros(1,2);
  for i=1:2
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
  scoreTable = table(lblNames', prec', recall', f1', 'VariableNames',...
                     {'Label Names','Precision','Recall', 'F1-Score'});
end


function rmse = RMSE(preds, lbls)
    rmse = sqrt(mean((preds-lbls).^2));
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