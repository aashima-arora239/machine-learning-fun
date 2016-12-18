load('train_dump.mat')
load('labels.mat')
load('test_dump.mat')
load('tslabels.mat')
load('tr_bigram.mat')
load('ts_bigram.mat');
load('test_dump.mat')
load('tslabels.mat')

%% Variables imported to MATLAB after preprocessing from Python.
%ts_matrix  - Reviews test matrix
%bow_matrix - Bag of words matrix for reviews
%tr_bigrm_mat - Training bigram matrix 
%ts_bigrm_mat =- Testing bigram matrix 
%labels - Reviews training labels
%tslabels -  Reviews test label matrix

num_docs = 200000;
labels = double(labels); 
tslabels = double(tslabels);
labels = labels((1:num_docs));
tslabels = tslabels((1:num_docs));

%% Perform Cross Validation on Data Set 

% % 5 Fold Cross Validation
k = 5;
X=['For N = ',num2str(num_docs),' and Folds = ',num2str(k)];
disp(X);
% 
% COMBINATIONS
% 1. Naive Bayes(Unigram)
% 2. Averaged Perceptron (Unigram, Tfidf, Bigram, tfIdf(log based))

% %NAIVE BAYES

%--Initilization---
tr_labels = labels;

%-----UNIGRAM------- %
tr_data = bow_matrix((1:num_docs),:);
% Calling crossvalidation 
disp('Performing Naive Bayes Classification for Unigram Model');
q2_crossval(k,tr_data, tr_labels,@naivebayes);

% 
% %AVERAGE PERCEPTRON

%--Initilization---
tr_labels = labels;
tr_labels(tr_labels == 0) = -1;
% 
%-----UNIGRAM------- %
tr_data = bow_matrix((1:num_docs),:);
% Calling crossvalidation
disp('Performing Perceptron Classification for Unigram Model');
q2_crossval(k,tr_data, tr_labels,@perceptron_classify);
% 
%----TFIDF-----------%
tr_data = bow_matrix((1:num_docs),:);
tr_data = construct_tfidf_mat_b(tr_data);

% Calling crossvalidation
disp('Performing Perceptron Classification for TFIDF Model');
q2_crossval(k,tr_data, tr_labels,@perceptron_classify);
% 
%----BIGRAM-----------%
tr_data = tr_bigrm_mat((1:num_docs),:);
% Calling crossvalidation
disp('Performing Perceptron Classification for BIGRAM Model');
q2_crossval(k,tr_data, tr_labels,@perceptron_classify);

% Since this is the best, perform testing on this%

ts_data = test_matrix((1:num_docs),:);
ts_labels = tslabels;
ts_labels(ts_labels == 0) = -1;
%Training Error %
q2_testing(tr_data, tr_labels, tr_data, tr_labels, @perceptron_classify);


%Testing Error%

%----TFIDF-LOG-----------%
tr_data = bow_matrix((1:num_docs),:);
tr_data = construct_tfidf_mat_d(tr_data);
% Calling crossvalidation
disp('Performing Perceptron Classification for LOG TFIDF Model');
q2_crossval(k,tr_data, tr_labels,@perceptron_classify);






