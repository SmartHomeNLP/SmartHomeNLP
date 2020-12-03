'''
02-12-2020: 
new document which checks models,
both in terms of performance and 
visual inspection. VMP.
'''
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
import NLP_visualization as NLP_vis
import clean_text as clean_fun
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
from gensim import corpora, models, utils
import pyLDAvis
import pyLDAvis.gensim


## load models, dictionary & corpus: 
with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\corpus.pkl', 'rb') as corpus:
    corpus = pickle.load(corpus)

with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\dict.pkl', 'rb') as dictionary:
    dictionary = pickle.load(dictionary)

with open('C:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\models\\models_mallet_init.pkl', 'rb') as mallet_mods:
    mallet = pickle.load(mallet_mods)

## coherence: 
## trying some of her stuff: 
os.getcwd()
df_train = pd.read_csv('../data/preprocessed/data_clean.csv')
texts = df_train['clean_text'].tolist()
texts

## u-mass for now - have to get c_v or something else:
## however, for now c_v does not work..
coherence_umass = []

for i in mallet.keys():
            print("Calculate Coherence gensim cv in {}...".format(i))
            coherencemodel = CoherenceModel(model = mallet[i], corpus = corpus, coherence = 'u_mass')
            coherence_umass.append(coherencemodel.get_coherence())

coherence_umass

## convert to gensim:
gensim_mods = {}
for i in mallet: 
    mod = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet[i])
    name = "{0}".format(i)
    gensim_mods[name] = mod

## perplexity: (not clear why this does not work)
for i in gensim_mods: 
    print('\nPerplexity: ', gensim_mods[i].log_perplexity(corpus))

## get the words out (from MALLET)
for i in mallet: 
    print(mallet[i].print_topics(5, 5))

## get security out: 
for i in gensim_mods: 
    print(f"{i}", gensim_mods[i].get_term_topics("security", minimum_probability=0.01))

## check it out more closely: 
#### for the one with 20 topics
gensim_mods["num_topics20"].show_topic(3, 10)
gensim_mods["num_topics20"].show_topic(12, 10)

#### for the one with 50 topics
gensim_mods["num_topics50"].show_topic(17, 10)
gensim_mods["num_topics50"].show_topic(35, 10)
gensim_mods["num_topics50"].show_topic(39, 10)

## pyLDAvis 
pyLDAvis.enable_notebook()
top20 = pyLDAvis.gensim.prepare(gensim_mods["num_topics20"], corpus, dictionary, sort_topics = True)
pyLDAvis.display(top20)

## word-clouds: 
## could be extended & made better.
import matplotlib.pyplot as plt
from wordcloud import WordCloud
for t in range(gensim_mods["num_topics20"].num_topics):
    plt.figure()
    plt.imshow(WordCloud().fit_words(dict(gensim_mods["num_topics20"].show_topic(t, 100))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
