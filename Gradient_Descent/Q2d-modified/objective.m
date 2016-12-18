function[value] = objective(w,tdata,tlabels) 
temp = sum(repmat(w,size(tdata,1),1).*tdata,2);
value = (log(1 + exp(temp)) - tlabels.*temp); 
value = sum(value)/size(tdata,1);
end