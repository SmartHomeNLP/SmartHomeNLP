import multiprocessing
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
#from gensim.models.wrappers import LdaMallet
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
import NLP_visualization as NLP_vis
import clean_text as clean_fun
from sklearn.feature_extraction.text import CountVectorizer
# set up logging so we see what's going on
import logging
import os
from gensim import corpora, models, utils
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

help(multiprocessing.Pool())
with multiprocessing.Pool()