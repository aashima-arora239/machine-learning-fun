load('news.mat')
%political = [17,18,19]; = +1
%religious = [1,16,20]  =  -1;
set = [17,18,19,1,16,20];
countp = 0;
countr = 0;
countpt = 0;
countrt = 0;
new_trdata = zeros(0,0);
new_tsdata = zeros(0,0);
for i=1:6
    V = find(labels == set(i));
    VT = find(testlabels == set(i));
    new_trdata = [new_trdata ; data(V,:)];
    new_tsdata = [new_tsdata ; testdata(VT,:)];
    if i == 3
        countp = size(new_trdata,1);
        countpt = size(new_tsdata,1);
    end
    if i == 6
        countr = size(new_trdata,1) - countp;
        countrt = size(new_tsdata,1) - countpt;
    end
end
newtr_labels = repelem(1,countp);
newtr_labels = [newtr_labels repelem(0,countr)];
newts_labels = repelem(1,countpt);
newts_labels = [newts_labels repelem(0,countrt)];

% Estimate pi(k) using laplace smoothing,
% Estimate there(i,j,k) using MLE 
% Finalize the label for the new vector x that maximizes the probability of
% the new vector approximating the label y.
tic;
num_features = size(new_trdata,2);
theta = zeros(2,num_features);
pi = zeros(2,1);
for y_label = 0:1
    pi(y_label+1) = sum(newtr_labels == y_label)/numel(newtr_labels);
    %theta(i,j,k) = sum where xi = xi,j and y=y_label / total examples
    %where y = y_label
    V = find(newtr_labels == y_label);
    subset = new_trdata(V,:);
    theta(y_label+1,:) = (sum(subset) + 1)/(size(subset,1) + 2);    
end

% For classifying the new vector x, maximize pi(y_label) x
% mul(theta(feature,label)) for all n and y being 1 to 20
weight = log(theta(2,:)) + log(1 - theta(1,:)) - log(theta(1,:)) - log(1 - theta(2,:));
bias = sum(log(1 - theta(2,:)) - log(1 - theta(1,:))) + log(pi(2)) - log(pi(1));
arg = new_tsdata*weight' + bias;
I = arg > 0;
values = (newts_labels' ~= I);
error_rate = sum(values)/size(newts_labels,2);
toc;
[sortedX,sortingIndices] = sort(weight,'descend');
maxValueIndices = sortingIndices(1:20);
minValueIndices = sortingIndices(end - 20 + 1:end);
A = importdata('news.vocab');
disp('Maximum');
disp(A(maxValueIndices,:));
disp('Minimum');
disp(wrev(A(minValueIndices,:)));
