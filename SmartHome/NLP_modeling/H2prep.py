## import stuff
import logging
import os
import time
import warnings
import copy
import re
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.matutils import corpus2csc, Sparse2Corpus
from gensim.models import LdaModel, LdaMulticore
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
from gensim import corpora, models, utils
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

## read stuff
os.getcwd()
data = pd.read_csv("../data/clean/H2submissions.csv")

# count vectorizer 
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)

X = bigram_vectorizer.fit_transform(data['clean_text'].tolist()) #.values.astype('U')

term2id = bigram_vectorizer.vocabulary_

# more stuff 
corpus = Sparse2Corpus(X, documents_columns=False)
dictionary = corpora.Dictionary.from_corpus(corpus, id2word= {v:k for (k, v) in term2id.items()})

## filter dictionary
filter_dict = copy.deepcopy(dictionary)
filter_dict.filter_extremes(no_below=5, no_above=0.4)

## something must be superfluous somewhere?
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict.items()})

train_bigram = bigram_vectorizer.fit(data['clean_text'].tolist())

## 
X = bigram_vectorizer.transform(data['clean_text'].tolist())

corpus = Sparse2Corpus(X, documents_columns=False)

## save stuff
os.getcwd()

# save the submissions
with open('../data/modeling/H2submissions.pkl', 'wb') as H2sub: 
    pickle.dump(data, H2sub) 

## save the corpus: 
with open('../data/modeling/H2corpus.pkl', 'wb') as H2corpus:
    pickle.dump(corpus, H2corpus)

## save the dictionary: 
with open('../data/modeling/H2dictionary.pkl', 'wb') as H2dictionary:
    pickle.dump(filter_dict, H2dictionary) #filter_dict

