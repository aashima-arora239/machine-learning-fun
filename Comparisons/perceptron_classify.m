function [error_rate] = perceptron_classify(tr_data, ts_data, tr_labels, ts_labels)
num_rows = size(tr_data,1);
weight = tr_data(1,:);
bias = 0;
tic;
for i=1:2
    I = randperm(num_rows);
    tr_data = tr_data(I,:);
    tr_labels = tr_labels(I);
    weight_avg = weight;
    for t=2:num_rows
        adjust = tr_labels(t)*(dot(tr_data(t,:),weight) + bias); %y(wx + b)
        if adjust <= 0
            weight = weight + tr_labels(t)*tr_data(t,:);
            bias = bias + tr_labels(t);
        end
        if i == 2
            weight_avg = weight_avg + weight; % Averaging only on second iteration
        end
    end
end
toc;
weight_avg = weight_avg/(num_rows + 1); % Averaging on n+1 iterations
final = ts_data* weight_avg(1:size(ts_data,2))' + bias;
unmatched = (ts_labels ~= sign(final'));
error_rate = sum(unmatched)/size(ts_data,1);
end


