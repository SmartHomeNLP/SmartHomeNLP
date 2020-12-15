## perhaps figure out what we can leave out
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import pickle
import numpy as np 
import time
import logging
import gensim
from gensim.models import LdaModel, LdaMulticore
import multiprocessing
from multiprocessing import Process, freeze_support
import threading
import timeit
import functools
import logging
import warnings
import copy
import re
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import pickle
import pandas as pd
import seaborn as sns
from gensim.matutils import corpus2csc, Sparse2Corpus
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, utils
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# timer 
def timer(func): 
    """Prints run-time""" 
    @functools.wraps(func)

    def wrapper_timer(*args, **kwargs): 
        
        ## start time
        start_time = timeit.default_timer()

        ## print that you start
        print(f"Starting {func,__name__!r}")

        ## generate the value from the function
        value = func(*args, **kwargs) 
        
        ## end time 
        end_time = timeit.default_timer() 

        ## calculate run-time
        run_time = (end_time - start_time)/60 

        ## print statement 
        print(f"Finished {func,__name__!r} in {run_time:.4f} minutes") 
        
        return value 
    return wrapper_timer

## GEN. CORPUS & DICTIONARY
@timer
def corpus_dct_gen(names = [], done = False): 

    if done: 
        print("skipped bot cleaning")
        return(None)

    for i in names: 
        with open(f"../data/clean/{i}.pkl", 'rb') as f: 
            data = pickle.load(f) 
        
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

        ## save the dictionary: 
        with open(f'../data/modeling/{i}_dct.pkl', 'wb') as dictionary:
            pickle.dump(filter_dict, dictionary) #filter_dict

        ## save the corpus: 
        with open(f'../data/modeling/{i}_corpus.pkl', 'wb') as c:
            pickle.dump(corpus, c)

## RUN MULTICORE
@timer
def LdaMulti_topics(dictionary, corpus, limit, start, step, alpha, eta, done = False):
    '''
    Parameters:
    ---------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    limit: max num of topics (end at limit -1)
    start: min number of topics
    step: increment between each integer
    alpha: list of prior on the per-document topic distribution.
    eta: list of prior on the per-topic word distribution. 

    Returns:
    ---------
    model_list: list of LDA topic
    '''

    if done: 
        print("skipped bot cleaning")
        return(None)

    # initialize an empty dictionary
    models = {}
    global_start = timeit.default_timer()
    for a in alpha: 
        print('Running model with alpha={}/k'.format(a))
        for b in eta:
            print('Running model with beta={}'.format(b))
            for num_topics in range(start, limit, step):
                print('Running model with number of topics (k): ', num_topics)
                start_time = timeit.default_timer()
                model = LdaMulticore(corpus=corpus, 
                id2word=dictionary,
                num_topics=num_topics, 
                random_state=123,
                passes=10, 
                alpha=[a]*num_topics, 
                workers = 7, #for a 4-core PC. 
                chunksize = 2000,
                eta=b,
                eval_every = None,
                per_word_topics=True)
                stop_time = timeit.default_timer()
                print(f'it took {(stop_time - start_time)/60} min')
                name = "a{0}_b{1}_k{2}".format(a, b, num_topics)
                models[name] = model
    global_end = timeit.default_timer()
    print(f'it took {(global_end - global_start)/60} minutes')

    return models