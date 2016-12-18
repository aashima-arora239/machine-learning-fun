load 'hw4data.mat'
rows = size(data,1);
range_vals = max(data) - min(data);
A = diag(1./range_vals);
data = (A*data')'; 
bias = ones(rows,1);
data = [data bias];
tic;
weight = logistic_gradient_descent(data,labels);
toc;
