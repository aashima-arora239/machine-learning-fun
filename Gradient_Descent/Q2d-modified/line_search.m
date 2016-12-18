function [learning_rate] = line_search(weight,gradient,tdata,tlabels)
learning_rate = 1;
alpha = weight - learning_rate*gradient;
while objective(alpha,tdata,tlabels) > (objective(weight,tdata,tlabels) - 0.5*learning_rate*(norm(gradient))^2)
    learning_rate = 0.5*learning_rate;
    alpha = weight - learning_rate*gradient;
end
end