% Part b of the question 
function [tfidf_mat] = construct_tfidf_mat_b(tf_mat)
num_docs = size(tf_mat,1);
tf_mat = tf_mat((1:num_docs),:); % Tf slice of the training data
numerator = repelem(1 + num_docs,size(tf_mat,2));
df_vector = 1 + sum(tf_mat > 0);
idf = log10(numerator./df_vector); % log based idf without adjustment for terms occuring in each doc
tfidf_mat = bsxfun(@times,tf_mat,idf);
end
