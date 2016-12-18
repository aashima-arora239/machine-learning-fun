function [error_rate] = naivebayes(tr_data, ts_data, tr_labels, ts_labels)
% Estimate pi(k) using laplace smoothing,
% Estimate there(i,j,k) using MLE 
% Finalize the label for the new vector x that maximizes the probability of
% the new vector approximating the label y.
tic;
num_features = size(tr_data,2);
theta = zeros(2,num_features);
pi = zeros(2,1);
for y_label = 0:1
    pi(y_label+1) = sum(tr_labels == y_label)/numel(tr_labels);
    %theta(i,j,k) = sum where xi = xi,j and y=y_label / total examples
    %where y = y_label
    V = find(tr_labels == y_label);
    subset = tr_data(V,:);
    theta(y_label+1,:) = (sum(subset) + 1)/(size(subset,1) + 2);    
end

% For classifying the new vector x, maximize pi(y_label) x
% mul(theta(feature,label)) for all n and y being 1 to 20
weight = log(theta(2,:)) + log(1 - theta(1,:)) - log(theta(1,:)) - log(1 - theta(2,:));
bias = sum(log(1 - theta(2,:)) - log(1 - theta(1,:))) + log(pi(2)) - log(pi(1));
arg = ts_data*weight(1:size(ts_data,2))' + bias;
I = arg > 0;
values = (ts_labels' ~= I);
error_rate = sum(values)/size(ts_labels,2);
toc;