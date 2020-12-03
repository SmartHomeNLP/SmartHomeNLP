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

# Filter out words that occur in less than 5 comments or more than 40% of 
# comments but don't cap it besides this. 380K
filter_dict_380K = copy.deepcopy(dictionary) 
filter_dict_380K.filter_extremes(no_below = 4, no_above = 0.4, keep_n = None)
NLP_vis.vocabulary_descriptive(filter_dict_380K, corpus) 

# SAVE DICTIONARY
'''
tmp_file = datapath('vocabulary\\test_nb5_na04')
filter_dict.save(tmp_file) ### NOTE: The file is relative to gensim and its test-data folder in your venv. Here you must create a new folder called "vocabulary" and "train_bigram"
'''

#Update corpus to the new dictionary
bigram_vectorizer_100K = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict_100K.items()})

train_bigram_100K = bigram_vectorizer_100K.fit(df_train['clean_text'].tolist())

bigram_vectorizer_380K = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict_380K.items()})

train_bigram_380K = bigram_vectorizer_380K.fit(df_train['clean_text'].tolist())

'''
# SAVE BIGRAM
filename = datapath('train_bigram\\nb5_na04_bigram.pkl')
with open(filename, "wb") as f:
    pickle.dump(train_bigram, f)

'''

## this has to happen again it seems:
X_100K = bigram_vectorizer_100K.transform(df_train['clean_text'].tolist())
X_380K = bigram_vectorizer_380K.transform(df_train['clean_text'].tolist())
corpus_100K = Sparse2Corpus(X_100K, documents_columns=False)
corpus_380K = Sparse2Corpus(X_380K, documents_columns=False)

NLP_vis.vocabulary_descriptive(filter_dict_100K, corpus_100K)
NLP_vis.vocabulary_descriptive(filter_dict_380K, corpus_380K)

'''

def mallet_fun(path, corpus, dictionary, start, step, alpha, eta):

    # initialize an empty dictionary
    models = {}

    for a in alpha: 
        print('Running model with alpha={}/k'.format(a))
        for b in beta:
            print('Running model with beta={}'.format(b))
            for num_topics in range(start, limit, step):
                print('Running model with number of topics (k): ', num_topics)
                model = LdaModel(corpus=corpus, id2word=dictionary,
                                     num_topics=num_topics, random_state=123, 
                                     passes=10, 
                                     alpha=[a]*num_topics, eta=b,
                                     per_word_topics=True)
                
                name = "a{0}_b{1}_k{2}".format(a, b, num_topics)
                models[name] = model

                print('\n')

    return models
'''

## MODELS (mallet)
mallet_path = 'C:/mallet/bin/mallet'

## timing it: 
import timeit

## manually one at a time: 14.24 minutes. 
start = timeit.default_timer()
print(f"starting corpus 100K")
mod_100K = LdaMallet(mallet_path, corpus_100K, num_topics = 10, id2word = filter_dict_100K, alpha = 20, optimize_interval = 10)
end = timeit.default_timer() 
print(f"{(end - start)/60} min. to end corpus 100K")

## the big one: 17.47 minutes. 
start = timeit.default_timer()
print(f"starting corpus 380K")
mod_380K = LdaMallet(mallet_path, corpus_380K, num_topics = 10, id2word = filter_dict_380K, alpha = 20, optimize_interval = 10)
end = timeit.default_timer() 
print(f"{(end - start)/60} min. to end corpus 380K")

## convert to gensim: 
gensim_100K = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mod_100K)
gensim_380K = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mod_380K)

## check the two models 
gensim_100K.print_topics(num_topics = 10, num_words = 10)
gensim_380K.print_topics(num_topics = 10, num_words = 10)

## visualize them pyLDAvis
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis_100K = pyLDAvis.gensim.prepare(gensim_100K, corpus_100K, filter_dict_100K, sort_topics = True)
pyLDAvis.display(vis_100K)

vis_380K = pyLDAvis.gensim.prepare(gensim_380K, corpus_380K, filter_dict_380K, sort_topics = True)
pyLDAvis.display(vis_380K) 

## coherence scores 
# u-mass: (fast, should be close to 0):
CM_100K = CoherenceModel(model = gensim_100K, corpus = corpus_100K, coherence='u_mass')
CM_100K_COH = CM_100K.get_coherence() 

CM_380K = CoherenceModel(model = gensim_380K, corpus = corpus_380K, coherence='u_mass')
CM_380K_COH = CM_380K.get_coherence() 

# 380K does not seem to be better but this is 
# a very shallow test. 
CM_380K_COH
CM_100K_COH

## coherence score c_v (multiprocessing issue)
def main():

    ## data 
    texts = df_train['clean_text'].tolist()

    ## models 
    ## should corpus be included? (not clear)
    cv_optim = CoherenceModel(model=model_optim, texts=texts, corpus = corpus, dictionary=filter_dict, coherence='c_v')
    cv_nonoptim = CoherenceModel(model=mod_gensim, texts=texts, corpus = corpus, dictionary=filter_dict, coherence='c_v')

    ## get coherence
    print(cv_nonoptim.get_coherence())
    print(cv_optim.get_coherence())

if __name__=='__main__':
    main()

## checking hyperparameters: 
model_optim.alpha
model_optim.print_topic(5)
model_optim.iterations
model_optim.topic_threshold
model_optim.eta
model.alpha
mod_gensim.eta

import pyLDAvis
import pyLDAvis.gensim

# Logging Gensim's output
# return time in seconds since the epoch
log_file = os.getcwd() + r'\logging' + r'\log_%s.txt' % int(time.time())

logging.basicConfig(filename=log_file,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
