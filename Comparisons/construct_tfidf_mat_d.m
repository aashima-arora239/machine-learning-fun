% Part d of the question 
function [tfidf_mat] = construct_tfidf_mat_d(tf_mat)
num_docs = size(tf_mat,1);
tf_mat = tf_mat((1:num_docs),:); % Slice of the tf matrix of the training data
numerator = repelem(1 + num_docs,size(tf_mat,2));
df_vector = 1 + sum(tf_mat > 0); % Document Frequency Vector
idf = 1 + log10(numerator./df_vector); % log based IDF 
tf = spfun(@log10,10*tf_mat); % eq to 1 + logtf
tfidf_mat = bsxfun(@times,tf,idf);
end


