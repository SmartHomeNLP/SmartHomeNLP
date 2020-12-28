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
with open("../data/models/H2_submissions_t5-50.pkl", "rb") as f: 
    H2_submissions = pickle.load(f)

# load corpus & dictionary: 
with open("../data/modeling/H2_submissions_corpus.pkl", "rb") as f:
    sub_corpus = pickle.load(f)

with open("../data/modeling/H2_submissions_dct.pkl", "rb") as f:
    sub_dct = pickle.load(f) 

## motivated by coherence measures: 
sub15 = H2_submissions['a0.01_b0.1_k15']
sub30 = H2_submissions['a0.01_b0.1_k30']

## duplicate function (clean up):
def topic_names(models, k): 

    liste = [] 

    for i in range(k): 
        print(models.print_topic(i, 10))
        label = input("Topic Label: ")
        liste.append(label)
        print(liste)

    return liste

## topic names: 
sub15_names = topic_names(sub15, 15) 
sub30_names = topic_names(sub30, 30) #

## write topic lists: 
dct = {"sub15_tags": sub15_names, "sub30_tags": sub30_names}

with open("../data/evaluation/H2_labels.pkl", "wb") as f:
    pickle.dump(dct, f)
