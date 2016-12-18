load 'hw4data.mat'
trset = floor(0.8*size(data,1));
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
toc;
fprintf('The final objective value is %f\n',objective(weight,trdata,trlabels));