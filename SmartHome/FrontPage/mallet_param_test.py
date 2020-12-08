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

mallet_path = 'C:/mallet/bin/mallet'
num_topics = 20
mallet_a1 = LdaMallet(mallet_path, corpus, num_topics = num_topics, alpha = 1 * num_topics, id2word = dictionary)

with open('../models/mallet_test_a1.pkl', 'wb') as mallet:
    pickle.dump(mallet_a1, mallet)