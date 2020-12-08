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

def main():

## load stuff 
    with open('../models/corpus.pkl', 'rb') as corpus:
        corpus = pickle.load(corpus)

    with open('../models/dict.pkl', 'rb') as dictionary:
        dictionary = pickle.load(dictionary)

    with open('../models/models_mallet_init.pkl', 'rb') as mallet_mods:
        mallet = pickle.load(mallet_mods)

    ## convert to gensim:
    gensim_mods = {}
    for i in mallet: 
        mod = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet[i])
        name = "{0}".format(i)
        gensim_mods[name] = mod

    ## convert to correct text format:
    def corpus2token_text(corpus, dictionary):
                nested_doc = []
                texts = []
                for doc in corpus:
                    nested_doc.append([[dictionary[k]]*v for (k, v) in doc])
                for doc in nested_doc:
                    texts.append([item for sublist in doc for item in sublist])
                return texts

    texts = corpus2token_text(corpus, dictionary)

    ## c_v 
    coherence_gensim_c_v = []
    for i in gensim_mods:
        print("Calculate Coherence gensim cv in {}...".format(i))
        coherencemodel = CoherenceModel(model = gensim_mods[i], texts = texts,
                                        dictionary = dictionary, coherence = 'c_v')
        coherence_gensim_c_v.append(coherencemodel.get_coherence())
    
    ## better format: 
    

    ## save the results in the evaluation folder: 
    with open('../evaluation/models_c_v.pkl', 'wb') as models_c_v:
        pickle.dump(coherence_gensim_c_v, models_c_v)

if __name__ == '__main__':
    #freeze_support()
    main()


'''
    for i in gensim_mods:
        print("Calculate Coherence gensim cv in {}...".format(i))
        coherencemodel = CoherenceModel(model = gensim_mods[i], texts = texts,
                                        dictionary = dictionary2, coherence = 'c_v')
        coherence_gensim_c_v.append(coherencemodel.get_coherence())
    '''