load 'housing.mat'

w = data\labels;
y = (testdata*w - testlabels).^2;
sq_error = sum(y)/size(testdata,1);
disp(sq_error);
data(:,1) = [];
testdata(:,1) = [];
[B, FitInfo] = lasso(data,labels,'Lambda',2.4);
new = testdata*B + FitInfo.Intercept ;
k = (new - testlabels).^2;
error = sum(k)/size(testdata,1);
disp(error);