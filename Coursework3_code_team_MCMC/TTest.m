
significance = 0.1;
printCTtestResult(significance);

significance = 0.05;
printCTtestResult(significance);

significance = 0.01;
printCTtestResult(significance);

significance = 0.1;
printRTtestResult(significance);

significance = 0.05;
printRTtestResult(significance);

significance = 0.01;
printRTtestResult(significance);

function printCTtestResult(significance)
    cann = [0.98578, 1.00000, 1.00000, 0.99526, 0.99052, 1.00000, 0.99526, 0.99526, 0.99052, 1.00000];
    cdt = [0.99526, 0.99526, 0.99526, 0.99526, 0.98578, 0.99526, 0.99052, 0.99052, 0.97156, 0.98104];
    svm_linear = [0.99526, 0.99526, 0.99526, 0.99526, 0.99526, 0.98104, 0.99052, 1.00000, 0.99526, 0.99526];
    svm_rbf = [1.00000, 0.99526, 0.99526, 1.00000, 0.99526, 0.98578, 0.99526, 1.00000, 0.99052, 0.99526];
    svm_polynomial = [0.99052, 0.99526, 0.99526, 0.99526, 0.99526, 0.98578, 0.99052, 1.00000, 0.98104, 0.99526];
    fprintf(1, strcat("For Classification\nAt ", num2str(significance), " significance level:\n"));
    fprintf(1, strcat("ANN<==>DT have statistical difference:                      ", num2str(ttest2(cann, cdt, significance)), "\n"));
    fprintf(1, strcat("ANN<==>SVM(linear) have statistical difference:             ", num2str(ttest2(cann, svm_linear, significance)), "\n"));
    fprintf(1, strcat("ANN<==>SVM(rbf) have statistical difference:                ", num2str(ttest2(cann, svm_rbf, significance)), "\n"));
    fprintf(1, strcat("ANN<==>SVM(polynomial) have statistical difference:         ", num2str(ttest2(cann, svm_polynomial, significance)), "\n"));
    fprintf(1, strcat("DT<==>SVM(linear) have statistical difference:              ", num2str(ttest2(cdt, svm_linear, significance)), "\n"));
    fprintf(1, strcat("DT<==>SVM(rbf) have statistical difference:                 ", num2str(ttest2(cdt, svm_rbf, significance)), "\n"));
    fprintf(1, strcat("DT<==>SVM(polynomial) have statistical difference:          ", num2str(ttest2(cdt, svm_polynomial, significance)), "\n"));
    fprintf(1, strcat("SVM(linear)<==>SVM(rbf) have statistical difference:        ", num2str(ttest2(svm_linear, svm_rbf, significance)), "\n"));
    fprintf(1, strcat("SVM(linear)<==>SVM(polynomial) have statistical difference: ", num2str(ttest2(svm_linear, svm_polynomial, significance)), "\n"));
    fprintf(1, strcat("SVM(rbf)<==>SVM(polynomial) have statistical difference:    ", num2str(ttest2(svm_rbf, svm_polynomial, significance)), "\n"));
    fprintf(1, "\n");
end

function printRTtestResult(significance)
    rann = [7.0891, 1.8705, 1.7578, 3.0385, 3.4273, 2.6827, 4.8569, 3.5200, 7.7901, 3.0415];
    rdt = [3.0532, 3.3408, 2.8430, 3.3064, 3.1053, 2.6267, 2.5074, 2.9380, 3.0618, 2.8826];
    svr_linear = [4.9571, 4.8729, 5.2615, 3.9697, 4.9344, 4.7787, 5.3142, 4.9036, 4.9739, 4.8726];
    svr_rbf = [2.8420, 4.0759, 4.8601, 3.3412, 3.7048, 4.6006, 2.9508, 3.8349, 4.3221, 4.4890];
    svr_polynomial = [4.2796, 3.8930, 3.0814, 3.1251, 4.3978, 4.1812, 3.8979, 4.7270, 4.2715, 3.1137];
    fprintf(1, strcat("For Regression\nAt ", num2str(significance), " significance level:\n"));
    fprintf(1, strcat("ANN<==>DT have statistical difference:                      ", num2str(ttest2(rann, rdt, significance)), "\n"));
    fprintf(1, strcat("ANN<==>SVR(linear) have statistical difference:             ", num2str(ttest2(rann, svr_linear, significance)), "\n"));
    fprintf(1, strcat("ANN<==>SVR(rbf) have statistical difference:                ", num2str(ttest2(rann, svr_rbf, significance)), "\n"));
    fprintf(1, strcat("ANN<==>SVR(polynomial) have statistical difference:         ", num2str(ttest2(rann, svr_polynomial, significance)), "\n"));
    fprintf(1, strcat("DT<==>SVR(linear) have statistical difference:              ", num2str(ttest2(rdt, svr_linear, significance)), "\n"));
    fprintf(1, strcat("DT<==>SVR(rbf) have statistical difference:                 ", num2str(ttest2(rdt, svr_rbf, significance)), "\n"));
    fprintf(1, strcat("DT<==>SVR(polynomial) have statistical difference:          ", num2str(ttest2(rdt, svr_polynomial, significance)), "\n"));
    fprintf(1, strcat("SVR(linear)<==>SVR(rbf) have statistical difference:        ", num2str(ttest2(svr_linear, svr_rbf, significance)), "\n"));
    fprintf(1, strcat("SVR(linear)<==>SVR(polynomial) have statistical difference: ", num2str(ttest2(svr_linear, svr_polynomial, significance)), "\n"));
    fprintf(1, strcat("SVR(rbf)<==>SVR(polynomial) have statistical difference:    ", num2str(ttest2(svr_rbf, svr_polynomial, significance)), "\n"));
    fprintf(1, "\n");
end




