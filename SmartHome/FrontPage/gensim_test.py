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
from multiprocessing import Process, freeze_support
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
from sys import argv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('../models/corpus.pkl', 'rb') as corpus:
    corpus = pickle.load(corpus)

with open('../models/dict.pkl', 'rb') as dictionary:
    dictionary = pickle.load(dictionary)

### trying her stuff: 
num_topics = 20 
a = 1 
b = 1 
gensim_model = LdaModel(corpus=corpus, id2word=dictionary,
                                     num_topics=num_topics, random_state=123, 
                                     passes=10, 
                                     alpha=[a]*num_topics, eta=b,
                                     per_word_topics=True)

with open('../models/gensim_test.pkl', 'wb') as gensim:
    pickle.dump(gensim_model, gensim)