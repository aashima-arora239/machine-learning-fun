function[updated_weight] =  logistic_gradient_descent(training_data, training_labels)
DESIRED_VALUE= 0.65064;
cols = size(training_data,2);
rows = size(training_data,1);
updated_weight = zeros(1,cols);
count = 0;
objective_value = objective(updated_weight,training_data, training_labels);
while (objective_value >= DESIRED_VALUE)
    temp = sum(repmat(updated_weight,rows,1).*training_data,2);
    p1 = exp(temp);
    temp = p1./(1 + p1);
    gradient = repmat(temp,1,cols).*training_data - (repmat(training_labels,1,cols)).*training_data;
    gradient = sum(gradient)/rows;
    learning_rate = line_search(updated_weight,gradient,training_data, training_labels);
    updated_weight = updated_weight - learning_rate*gradient;
    objective_value = objective(updated_weight,training_data, training_labels);
    count = count+1;
end
fprintf('The number of iterations are %d to reach objective value %f\n',count,objective_value);
end