# https://senderle.github.io/topic-modeling-tool/documentation/2018/09/27/optional-settings.html

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
from gensim.models.wrappers import LdaMallet #easier?
from gensim.models.coherencemodel import CoherenceModel #needed
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
import NLP_visualization as NLP_vis
import clean_text as clean_fun
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
from gensim import corpora, models, utils
import pyLDAvis
import pyLDAvis.gensim

## logging and warnings 
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df_train = pd.read_csv('../data/preprocessed/data_clean.csv', encoding="utf-8")
NLP_vis.words_count(df_train["clean_text"])

# get source comments for further investigations
# comments = data.comments

# Train Bigram Models
# ignore terms that appeared in less than 2 documents
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)

X = bigram_vectorizer.fit_transform(df_train['clean_text'].tolist())

term2id = bigram_vectorizer.vocabulary_ #now becomes dict

# get gensim dictionary
# https://stackoverflow.com/questions/21552518/using-scikit-learn-vectorizers-and-vocabularies-with-gensim
# transform sparse matrix into gensim corpus: Term Document Frequency (id, freq) for each text
corpus = Sparse2Corpus(X, documents_columns=False)
dictionary = corpora.Dictionary.from_corpus(corpus, id2word= {v:k for (k, v) in term2id.items()})

# Words used in how many texts?
NLP_vis.vocabulary_descriptive(dictionary, corpus)

# Filter out words that occur less than 5 comments, or more than 40% of comments
# and cap it at 100,000 tokens. 
filter_dict_100K = copy.deepcopy(dictionary)
filter_dict_100K.filter_extremes(no_below=5, no_above=0.4) 
NLP_vis.vocabulary_freq_words(filter_dict_100K, False, 30)
NLP_vis.vocabulary_freq_words(filter_dict_100K, True, 30)
NLP_vis.vocabulary_descriptive(filter_dict_100K, corpus)

type(filter_dict_100K)
all_words = ' '.join([text for text in filter_dict_100K])


#Update corpus to the new dictionary
bigram_vectorizer_100K = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict_100K.items()})

train_bigram_100K = bigram_vectorizer_100K.fit(df_train['clean_text'].tolist())

## this has to happen again it seems:
X_100K = bigram_vectorizer_100K.transform(df_train['clean_text'].tolist())
corpus_100K = Sparse2Corpus(X_100K, documents_columns=False)

NLP_vis.vocabulary_descriptive(filter_dict_100K, corpus_100K)


## MODELS (mallet)
mallet_path = 'C:/mallet/bin/mallet'

## timing it: 
import timeit

## function to run: 
num_topics = [5, 10, 20, 50]

## does it automatically divide alpha by k as stated somewhere?
## what is a good value of alpha?
## how much (if at all) should we tune?
def MALLET_topics(dictionary, corpus, num_topics):
    models = {}

    for k in num_topics: 
        start = timeit.default_timer()
        print(f"starting with {k} topics")
        mod = LdaMallet(mallet_path, corpus, num_topics = k, id2word = dictionary, optimize_interval = 20)
        name = "num_topics{0}".format(k)
        models[name] = mod
        end = timeit.default_timer() 
        print(f"{(end - start)/60} min. with {k} topics")

    return models 

models = MALLET_topics(filter_dict_100K, corpus_100K, num_topics)

## topics for the different models: 
models['num_topics5'].print_topics(5, 10)
models['num_topics10'].print_topics(10, 10)
models['num_topics20'].print_topics(20, 10)
models['num_topics50'].print_topics(50, 10)

## save the models & all the other shit (corpus, library) ## 
models
os.getcwd()

## pickle saving corpus
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\corpus.pkl', 'wb') as corpus:
    pickle.dump(corpus_100K, corpus)

## open it again 
'''
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\corpus.pkl', 'rb') as corpus:
    corpus_test = pickle.load(corpus)

print(corpus_100K == corpus_test) ## False: but seems like the same?
corpus_100K
'''
## pickle saving dictionary
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\dict.pkl', 'wb') as dictionary:
    pickle.dump(filter_dict_100K, dictionary)

## open it again 
'''
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\dict.pkl', 'rb') as dictionary:
    dictionary_test = pickle.load(dictionary)

print(filter_dict_100K == dictionary_test) ## True: that is good. 
'''

## pickle saving models: 
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\models_mallet_init.pkl', 'wb') as mallet_mods:
    pickle.dump(models, mallet_mods)

'''
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\models_mallet_init.pkl', 'rb') as mallet_mods:
    mallet_test = pickle.load(mallet_mods)

print(models == mallet_test) ## False: but seems like the same?
'''


