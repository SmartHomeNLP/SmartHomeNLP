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

from evaluation_functions import *

# load models:
H1_tree = pickle_load("H1_tree_models", "models") 
H1_threadBIG = pickle_load("H1_thread_models_b0.1_a0.01_50-100", "models")
H1_threadSMALL = pickle_load("H1_thread_models_b0.1_b1", "models") 
thread_corpus = pickle_load("H1_thread_corpus", "modeling")
thread_dct = pickle_load("H1_thread_dct", "modeling")
tree_corpus = pickle_load("H1_tree_corpus", "modeling")
tree_dct = pickle_load("H1_tree_dct", "modeling")

# topics 30 & 100
tree30 = H1_tree['a0.01_b0.1_k30']
tree100 = H1_tree['a0.01_b0.1_k100']
thread30 = H1_threadSMALL['a0.01_b0.1_k30']
thread100 = H1_threadBIG['a0.01_b0.1_k100']
thread10 = H1_threadSMALL['a0.01_b0.1_k10']

## 
thread30.print_topics(30, 10)

## testing tree vs. threads: 
tree30_labels = topic_names(tree30, 30) #okay
thread30_labels = topic_names(thread30, 30) #good (we have amazon, apple, insteon as specific topics)

## these are not quite as promising. 
thread100_labels = topic_names(thread100, 30) #oddly specific and un-interpretable.
thread10_labels = topic_names(thread10, 10) #probably too broad. 

# save labels for tree vs. thread comparison: 
labels = {"tree30": tree30_labels, "thread30": thread30_labels}

with open("../data/evaluation/H1_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

#### argue qualitatively #### 

## tree vs. thread (30 vs 30) ##
# 1. qualitative overlap between tree & thread (how to & duplicates)
# 2. several in same category but split by companies. 
# 3. a couple of non-sense topics from tree to argue our point. 

## 100 vs. 30 vs. 10 ## 
# 1. 100 is non-sense and oddly specific. 
# 2. 10 is very overall & general.
# 3. 30 is pretty specific but meaningful. 

