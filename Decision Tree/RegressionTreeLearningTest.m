function tree = RegressionTreeLearningTest(ftrs, lbls, maxDepth, minRss, minSamples)
    ftrNames = ["Freq","Angle","Chord","FS_Velocity", "SSD_thickness"];
    stopCond = (length(unique(lbls))==1) || length(lbls)<=minSamples...
                || maxDepth == 1 || RSS(lbls)<=minRss;
    if stopCond 
        tree = rTree("", {}, mean(lbls), nan, nan, length(lbls)); 
    else
        [bestAtr, bestThld] = rChooseAttribute(ftrs, lbls);
        [lData, rData] = splitData([ftrs lbls], bestAtr, bestThld);
        lSubtree = RegressionTreeLearningTest(lData(:, 1:end-1), lData(:, end), maxDepth-1, minRss, minSamples);
        rSubtree = RegressionTreeLearningTest(rData(:, 1:end-1), rData(:, end), maxDepth-1, minRss, minSamples);
        op = strcat(ftrNames{bestAtr}, "<=", string(bestThld));
        tree = rTree(op, {lSubtree rSubtree}, nan, bestAtr, bestThld, length(lbls));
    end
end

function rss = RSS(lbls)
    rss = sum((lbls-mean(lbls)).^2);
end

function [eltData, gtData] = splitData(data, atr, thld)
    eltData = data(data(:, atr)<=thld, :);
    gtData = data(data(:, atr)>thld, :);
end

function [bestAtr, bestThld] = rChooseAttribute(ftrs, lbls)
    tmpData = [ftrs lbls];
    minRss = inf;    
    for i=1:size(ftrs, 2)
        thlds = ftrs(:, i);
        thlds = sortrows(thlds);
        for j=1:size(thlds, 1)-1
            [eltData, gtData] = splitData(tmpData, i, thlds(j, 1));
            rss = RSS(eltData(:, end)) + RSS(gtData(:, end));
            if rss < minRss
                minRss = rss;
                bestAtr = i;
                bestThld = thlds(j, 1);
            end
        end
    end
end

function tree = rTree(op, kids, pred, attr, thre, samp)
    tree.op = op;
    tree.kids = kids;
    tree.prediction = pred;
    tree.attribute =  attr;
    tree.threshold =  thre;
    tree.samples = samp;
end