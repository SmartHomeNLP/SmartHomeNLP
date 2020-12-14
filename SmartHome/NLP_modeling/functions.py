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

def LdaMulti_topics(dictionary, corpus, limit, start, step, alpha, eta):
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
                workers = 3, #for a 4-core PC. 
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