import pandas as pd 
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns 
from os import listdir
from os.path import isfile, join
import re 
import gensim 
import pyLDAvis
import pyLDAvis.gensim
import warnings
from wordcloud import WordCloud
warnings.filterwarnings("ignore", category=DeprecationWarning)

##### import functions #######
from visualization_functions import * 

###### pyLDAvis (reproducing our way of partialling it up?)  ######
## read corpus and dictionary 
thread_dct = pickle_load("H1_thread_dct", "modeling")
thread_corpus = pickle_load("H1_thread_corpus", "modeling")
tree_dct = pickle_load("H1_tree_dct", "modeling")
tree_corpus = pickle_load("H1_thread_corpus", "modeling")

#### next part takes some time ####
## first visualize thread 
## this looks absolutely terrible (hmm)
pyLDAvis.enable_notebook()
thread_pyLDA = pyLDAvis.gensim.prepare(thread, thread_corpus, thread_dct, sort_topics = False) ## so that it corresponds to gensim
pyLDAvis.display(thread_pyLDA)

## then visualize tree 
tree_pyLDA = pyLDAvis.gensim.prepare(tree, tree_corpus, tree_dct, sort_topics = False)
pyLDAvis.display(tree_pyLDA)