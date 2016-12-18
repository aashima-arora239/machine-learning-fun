% This function performs Cross Validation on the training data 
function[] = q2_crossval(k,tr_data, tr_labels,classifier_handle)
% Training on 200000 sets
num_docs = size(tr_data,1);
c = cvpartition(num_docs,'kfold',k);
err = zeros(c.NumTestSets,1);
for i = 1:c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);
    err(i) = classifier_handle(tr_data(trIdx,:), tr_data(teIdx,:), tr_labels(trIdx), tr_labels(teIdx));
    disp(err(i));
end
cvErr = sum(err)/sum(c.TestSize);
X = ['CROSS VALIDATION ERROR IS -',num2str(cvErr)];
disp(X);
end