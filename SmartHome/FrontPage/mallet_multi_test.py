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
filter_dict = copy.deepcopy(dictionary)
filter_dict.filter_extremes(no_below=4, no_above=0.4) 
NLP_vis.vocabulary_freq_words(filter_dict, False, 30)
NLP_vis.vocabulary_freq_words(filter_dict, True, 30)
NLP_vis.vocabulary_descriptive(filter_dict, corpus)

# SAVE DICTIONARY
tmp_file = datapath('vocabulary\\test_nb5_na04')
filter_dict.save(tmp_file) ### NOTE: The file is relative to gensim and its test-data folder in your venv. Here you must create a new folder called "vocabulary" and "train_bigram"

#Update corpus to the new dictionary
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict.items()})

train_bigram = bigram_vectorizer.fit(df_train['clean_text'].tolist())

# SAVE BIGRAM
filename = datapath('train_bigram\\nb5_na04_bigram.pkl')
with open(filename, "wb") as f:
    pickle.dump(train_bigram, f)

X = bigram_vectorizer.transform(df_train['clean_text'].tolist())

corpus = Sparse2Corpus(X, documents_columns=False)
NLP_vis.vocabulary_descriptive(filter_dict, corpus)

## MODELS

mallet_path = 'C:/mallet/bin/mallet'
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
## timing it: 
import timeit

## gives a weird error for the last model. 
def mallet_fun(path, corpus, dictionary, workers):
    mods = {}
    for i in workers: 
        start = timeit.default_timer() 
        print(f"starting with workers {i}")
        model = models.wrappers.LdaMallet(path, corpus, num_topics = 10, id2word = dictionary, alpha = 20, workers = i)
        name = "workers{0}".format(workers)
        end = timeit.default_timer() 
        print(f"it took {(end-start)/60} minutes to run worker {i}")
        mods[name] = model 
    return mods

mods = mallet_fun(path = mallet_path, corpus = corpus, dictionary = filter_dict, workers = [4, -1])

## running with 4 workers 
start = timeit.default_timer() 
print(f"starting with workers 4")
model = models.wrappers.LdaMallet(mallet_path, corpus, num_topics= 10, id2word=filter_dict, alpha=20, workers = 4)
end = timeit.default_timer() 
print(f"it took {(end-start)/60} minutes to run worker 4")

## running with optimize_interval 
start = timeit.default_timer() 
print(f"starting with workers 4")
model_optim = models.wrappers.LdaMallet(mallet_path, corpus, num_topics= 10, optimize_interval = 10, id2word=filter_dict, alpha=20, workers = 4)
end = timeit.default_timer() 
print(f"it took {(end-start)/60} minutes to run worker 4")
model_optim.print_topcis

## check the two models 
model.print_topics(num_topics = 10, num_words = 10)
model_optim.print_topics(num_topics = 10, num_words = 10)

## convert to gensim: 
mod_gensim = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
mod_optim = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_optim)
mod_optim

## coherence scores. 
## consider which coherence score.. (u_mass should be fast):
mod_cm = CoherenceModel(model=mod_gensim, corpus=corpus, coherence='u_mass')
mod_coherence = mod_cm.get_coherence() 

mod_optim_cm = CoherenceModel(model=model_optim, corpus=corpus, coherence='u_mass')
opt_coherence = mod_optim_cm.get_coherence() 

mod_coherence
opt_coherence

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
