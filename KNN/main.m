
% Define a function that takes X,Y,testdata as arguments and returns preds
%Write a function that implements the 1-nearest neighbor classifier with 
%Euclidean distance. Your function should take as input a matrix of 
%training feature vectors X and a vector of the corresponding labels Y, 
%as well as a matrix of test feature vectors test. 
%The output should be a vector of predicted labels preds for all the
%test points. 
%Draw n random points from data, together with their corresponding labels. 
%In MAT- LAB, use sel = randsample(60000,n) to pick the n random indices, and data(sel,:) and labels(sel) 
%to select the examples; 
%Error rate of classifier f on a set of labeled examples D: errD(f) := # of 
%(x,y) ? D such that f(x)/|D|
load 'ocr.mat'
imagesc(reshape(data(1,:),28,28)');
n = [1000,2000,4000,8000];
error_rate = zeros(1,size(n,1));
for i=1:4
    sel = randsample(60000,n(i));
    tr_data = data(sel,:);
    tr_labels = labels(sel);
    tic;
    preds = kNN_classifier(tr_data,tr_labels,testdata);
    toc
    values = (testlabels ~= preds);
    error_rate(i) = sum(values)/10000;
end

plot(n,error_rate);
