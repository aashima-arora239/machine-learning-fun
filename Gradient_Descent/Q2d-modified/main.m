load 'hw4data.mat'
rows = size(data,1);
range_vals = max(data) - min(data);
A = diag(1./range_vals);
data = (A*data')'; 
trset = floor(0.8*rows);
trdata = data(1:trset,:);
tsdata = data(trset + 1:end,:);
trlabels = labels(1:trset);
tslabels = labels(trset + 1:end);
trbias = ones(size(trdata,1),1);
tsbias = ones(size(tsdata,1),1);
trdata = [trdata trbias];
tsdata = [tsdata tsbias];
tic;
weight = logistic_gradient_descent(trdata,trlabels,tsdata, tslabels);
fprintf('The final objective value is %f\n',objective(weight,trdata, trlabels));
toc;
