load 'hw4data.mat'
bias = ones(size(data,1),1);
data = [data bias];
tic;
weight = logistic_gradient_descent(data,labels);
toc;
