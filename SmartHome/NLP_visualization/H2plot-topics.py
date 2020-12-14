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
from gensim.models.wrappers import LdaMallet #easier?
from gensim.models.coherencemodel import CoherenceModel #needed
from gensim.test.utils import datapath
from tmtoolkit.topicmod import evaluate, tm_lda
#import NLP_visualization as NLP_vis
#import clean_text as clean_fun
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
from gensim import corpora, models, utils
import pyLDAvis
import pyLDAvis.gensim
import warnings
from wordcloud import WordCloud
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('../data/modeling/H2corpus.pkl', 'rb') as corpus:
    corpus = pickle.load(corpus)

with open('../data/modeling/H2dictionary.pkl', 'rb') as dictionary:
    dictionary = pickle.load(dictionary)

with open('../data/modeling/H2models-b01.pkl', 'rb') as gensim_mods:
    H2models = pickle.load(gensim_mods)

## not ready: 
def plot_images(gensim_model, topics, columns = 5):
    
    total_img = topics
    import math
    rows = math.ceil(topics/columns)
    fig, axs = plt.subplots(rows, columns)
    
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img,columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
                axs[i,j].imshow(WordCloud().fit_words(dict(gensim_model.show_topic((i*columns)+j, 100))))
                axs[i,j].axis("off")
                axs[i,j].set_title("Topic #" + str((i*columns)+j))
        total_img -= columns
    fig.tight_layout()
    fig.set_size_inches(15, 15)
    return fig

## run the function on the best model: 
plot_images(H2models["a0.1_b0.01_k25"], 25, columns = 3)

## subset the best model(s): 
best_model = H2models['a0.1_b0.01_k25']
best_model2 = H2models['a0.01_b0.01_k25']

## find security: 
best_model.get_term_topics("security", minimum_probability = 0.001)
best_model2.get_term_topics("security", minimum_probability = 0.001)

## get topics out: 
liste = [12, 13]
for topic in liste: 
    print(best_model.show_topic(topic, 10))

liste = [4, 12, 13]
for topic in liste: 
    print(best_model2.show_topic(topic, 10))

## more plotting: 