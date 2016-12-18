function[preds]=  kNN_classifier(X,Y,test)
% X - Matrix of feature vectors
% Y - Vector of labelled data
% test - Matrix of feature vectors
%
TX = test*X.'; % Multiply the test and training matrices
TT = sum(test.^2,2);
XX = sum(X.^2,2);
sqTest = repmat(TT,1,size(X,1)); % Sum the testing rows and replicate across training data
sqTraining = repmat(XX',size(test,1),1); % Sum the training rows and replicate across test data
final_mat = sqrt(sqTest + sqTraining - 2*TX); % Add and take the sqrt to form element wise euclidean dist
[~,I] = min(final_mat.'); % get minimum indices 
preds = Y(I); % Return the label of the indices.


