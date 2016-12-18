function[updated_weight] =  logistic_gradient_descent(training_data, training_labels, test_data, test_labels)
cols = size(training_data,2);
rows = size(training_data,1);
updated_weight = zeros(1,cols);
count = 1;
hold_out_error = 1;
best_prev = hold_out_error;
%If this hold-out error rate is more than 0.99 times that of the best hold-out error
%rate previously computed, and the number of iterations executed is at least 32
%(which is somewhat of an arbitrary number), then stop.
while (true)
    temp = sum(repmat(updated_weight,rows,1).*training_data,2);
    p1 = exp(temp);
    temp = p1./(1 + p1);
    gradient = repmat(temp,1,cols).*training_data - (repmat(training_labels,1,cols)).*training_data;
    gradient = sum(gradient)/rows;
    learning_rate = line_search(updated_weight,gradient,training_data, training_labels);
    updated_weight = updated_weight - learning_rate*gradient;
    if(floor(log2(count)) == log2(count))
        I = test_data*updated_weight' > 0;
        values = (test_labels ~= I);
        if(hold_out_error < best_prev)
            best_prev = hold_out_error;
        end
        hold_out_error = sum(values)/size(test_labels,1);
        if(hold_out_error > 0.99*best_prev && count >= 32)
            break;
        end
    end
    count = count+1;
    
end
fprintf('The number of iterations are %d to reach hold out error %f\n',count,hold_out_error);
end