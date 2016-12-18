# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 06:22:42 2016

@author: aashimaarora
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:24:21 2016

@author: aashimaarora
"""

import pandas as pd
import numpy as np
from scipy import io
import scipy.sparse as sps
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
df = pd.read_csv('reviews_tr.csv',header=0,delimiter=',',nrows=200000)
df2 = pd.read_csv('reviews_te.csv',header=0,delimiter=',',nrows=200000)
reviews = df['text'].values
labels = df['label'].values
reviews_test = df2['text'].values;
test_labels = df2['label'].values

#This can be modified for each test and training reviews
bow_matrix = CountVectorizer(
                             analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',\
                             ngram_range = (2,2),\ # default for unigram
                             ).fit_transform(reviews)
                             
 #This was changed for each test and training set along with ngram values
io.savemat('tr_bigram.mat',mdict={'tr_bigrm_mat':bow_matrix.astype(np.int32)})

io.savemat('labels.mat',mdict={'labels':labels})
io.savemat('tslabels.mat',mdict={'tslabels':test_labels})



            
        
    


