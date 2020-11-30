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

df_train = pd.read_csv('../data/preprocessed/data_clean.csv', encoding="utf-8")
NLP_vis.words_count(df_train["clean_text"])

# get source comments for further investigations
# comments = data.comments

# Train Bigram Models
# ignore terms that appeared in less than 2 documents
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)

X = bigram_vectorizer.fit_transform(df_train['clean_text'].tolist())

term2id = bigram_vectorizer.vocabulary_

# get gensim dictionary
# https://stackoverflow.com/questions/21552518/using-scikit-learn-vectorizers-and-vocabularies-with-gensim
# transform sparse matrix into gensim corpus: Term Document Frequency (id, freq) for each text
corpus = Sparse2Corpus(X, documents_columns=False)
dictionary = corpora.Dictionary.from_corpus(corpus, id2word= {v:k for (k, v) in term2id.items()})

# Words used in how many texts?
NLP_vis.vocabulary_descriptive(dictionary, corpus)

# Filter out words that occur less than 5 comments, or more than 40% of comments
filter_dict = copy.deepcopy(dictionary)
filter_dict.filter_extremes(no_below=5, no_above=0.4) 
NLP_vis.vocabulary_freq_words(filter_dict, False, 30)

# SAVE DICTIONARY
tmp_file = datapath('vocabulary\\test_nb5_na04')
filter_dict.save(tmp_file) ### NOTE: The file is relative to gensim and its test-data folder in your venv. Here you must create a new folder called "vocabulary" and "train_bigram"

#Update corpus to the new dictionary
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                    vocabulary={term:id_ for (id_, term) in filter_dict.items()})

train_bigram = bigram_vectorizer.fit(df_train['clean_text'].tolist())

# SAVE BIGRAM
filename = datapath('train_bigram\\nb5_na04_bigram.pkl')
with open(filename, "wb") as f:
    pickle.dump(train_bigram, f)

X = bigram_vectorizer.transform(df_train['clean_text'].tolist())

corpus = Sparse2Corpus(X, documents_columns=False)
NLP_vis.vocabulary_descriptive(filter_dict, corpus)

## MODELS

mallet_path = 'C:/mallet/bin/mallet'

model = models.wrappers.LdaMallet(mallet_path, corpus, num_topics=20, id2word=filter_dict, alpha=20)

model.print_topics(num_topics=20, num_words=10)

import pyLDAvis
import pyLDAvis.gensim

# Logging Gensim's output
# return time in seconds since the epoch
log_file = os.getcwd() + r'\logging' + r'\log_%s.txt' % int(time.time())

logging.basicConfig(filename=log_file,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def LdaGensim_topics(dictionary, corpus, limit, start, step, alpha, beta):
    '''
    Parameters:
    ---------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    limit: max num of topics (end at limit -1)
    start: min number of topics
    step: increment between each integer
    alpha: list of prior on the per-document topic distribution.
    beta: list of prior on the per-topic word distribution. 

    Returns:
    ---------
    model_list: list of LDA topic
    '''

    # initialize an empty dictionary
    models = {}

    for a in alpha: 
        print('Running model with alpha={}/k'.format(a))
        for b in beta:
            print('Running model with beta={}'.format(b))
            for num_topics in range(start, limit, step):
                print('Running model with number of topics (k): ', num_topics)
                model = LdaModel(corpus=corpus, id2word=dictionary,
                                     num_topics=num_topics, random_state=123, 
                                     passes=10, 
                                     alpha=[a]*num_topics, eta=b,
                                     per_word_topics=True)
                
                name = "a{0}_b{1}_k{2}".format(a, b, num_topics)
                models[name] = model

                print('\n')

    return models

### COMPARE GENSIM WITH MALLET:
##____________________________

# models = LdaGensim_topics(dictionary=filter_dict, corpus=corpus, 
                                                         #start=10, limit=11, step=5, 
                                                         #alpha = [50], 
                                                         #beta = [0.01])

# her_model = models.get("a50_b0.01_k10")
# her_model.print_topics(10, 10)
# model.print_topics(10,10)

### EMIL CODE:

#convert mallet model to lda
model_gensim = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
#Visualize with LDAVis

model_gensim.save("mallet_model") #save the model

model_gensim = LdaModel.load("mallet_model") #load the model

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model_gensim, corpus, filter_dict, sort_topics = False)
pyLDAvis.display(vis)

model_gensim.get_term_topics('security', minimum_probability=0.001)
model_gensim.get_term_topics('trust', minimum_probability=0.001)
model_gensim.get_term_topics('privacy', minimum_probability=0.001)

model_gensim.print_topics(20)

#print('\nPerplexity:', model_gensim.log_perplexity(corpus))

from gensim.models import CoherenceModel

#coherence_score_lda = CoherenceModel(model=model_gensim, texts=corpus, dictionary=filter_dict, coherence='c_v')
#coherence_score = coherence_score_lda.get_coherence()

#print('\nCoherence Score:', coherence_score)

### CHECK PREPROCESSING:
## Spacy lemmatization on all of the sentence
## Allowed pos_tags should lemmatize if possible and just include the word if it doesn't know it 


##____________________________


# [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50] triggered insufficient memory on RDP server.
#time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1585905944))

# SAVE MODELS
# Write pickles incrementally to a file 
# One pickles equal to a combination of alpha beta across all number of topics)

# Divide by groups of 40 (num of topics)
num = 0
for i in range(40, len(models)+40, 40):
    keys_list = list(models.keys())[num:i]
    tmp_file = datapath('train_models\\nb5_na04_{}models.pkl').format(re.sub(r"\.", "", re.findall(".*_", keys_list[0])[0]))
    with open(tmp_file, "wb") as handle:
        pickle.dump({k: v for k, v in models.items() if k in keys_list}, handle)
    num = i


