function[] = q2_testing(tr_data, tr_labels, ts_data, ts_labels, classifier_handle)
% Training and testing on 200000 sets
error_rate = classifier_handle(tr_data, ts_data, tr_labels,ts_labels);
X = ['The error rate is - ', num2str(error_rate)];
disp(X)
end
