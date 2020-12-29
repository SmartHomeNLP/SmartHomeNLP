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

## H1 (thread & tree)

# load models: 
with open("../data/models/H1_tree_models.pkl", "rb") as f: 
    H1_tree = pickle.load(f)

with open("../data/models/H1_thread_models_b0.1_a0.01_50-100.pkl", "rb") as f: 
    H1_threadBIG = pickle.load(f)

with open("../data/models/H1_thread_models_b0.1_b1.pkl", "rb") as f: 
    H1_threadSMALL = pickle.load(f) 

# load corpus & dictionary: 
with open("../data/modeling/H1_thread_corpus.pkl", "rb") as f:
    thread_corpus = pickle.load(f)

with open("../data/modeling/H1_thread_dct.pkl", "rb") as f:
    thread_dct = pickle.load(f) 

with open("../data/modeling/H1_tree_corpus.pkl", "rb") as f:
    tree_corpus = pickle.load(f) 

with open("../data/modeling/H1_tree_dct.pkl", "rb") as f: 
    tree_dct = pickle.load(f)

# topics 30 & 100
tree30 = H1_tree['a0.01_b0.1_k30']
tree100 = H1_tree['a0.01_b0.1_k100']
thread30 = H1_threadSMALL['a0.01_b0.1_k30']
thread100 = H1_threadBIG['a0.01_b0.1_k100']
thread10 = H1_threadSMALL['a0.01_b0.1_k10']

# print topics raw
def topic_names(models, k): 

    liste = [] 

    for i in range(k): 
        print(models.print_topic(i, 10))
        label = input("Topic Label: ")
        liste.append(label)
        print(liste)

    return liste

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

