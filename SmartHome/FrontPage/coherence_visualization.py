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


with open('../evaluation/models_c_v.pkl', 'rb') as model_cv:
    model_cv = pickle.load(model_cv)

with open('../evaluation/models_gensim_test.pkl', 'rb') as model_gen: 
    model_gen = pickle.load(model_gen)

## very bad values: 
model_cv
model_gen #much better, try to visualize (is it also visually better?)