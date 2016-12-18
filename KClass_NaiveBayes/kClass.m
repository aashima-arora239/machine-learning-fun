load('news.mat')

% Estimate pi(k) using laplace smoothing,
% Estimate there(i,j,k) using MLE 
% Finalize the label for the new vector x that maximizes the probability of
% the new vector approximating the label y.
tic;
num_features = size(data,2);
theta = zeros(20,num_features);
pi = zeros(20,1);
% Parameter estimation of class prior and class conditional distributions%
for y_label = 1:20
    pi(y_label) = sum(labels == y_label)/numel(labels);
    %theta(i,j,k) = sum where xi = xi,j and y=y_label / total examples
    %where y = y_label
    V = find(labels == y_label);
    subset = data(V,:);
    theta(y_label,:) = (sum(subset) + 1)/(size(subset,1) + 2);    
end

% For classifying the new vector x, maximize pi(y_label) x
% mul(theta(feature,label)) for all n and y being 1 to 20
weight = log(theta) - log(1 - theta);
bias = sum(log(1 - theta),2) + log(pi);
X1 = testdata*weight';
vec = repmat(bias,1,size(X1,1));
arg = X1 + vec';
[~,I] = max(arg,[],2);
values = (testlabels ~= I);
error_rate = sum(values)/size(testlabels,1);
toc;







