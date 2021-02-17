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

## load function. 
def pickle_load(filename, folder): 
    with open(f"../data/{folder}/{filename}.pkl", "rb") as f:
        file = pickle.load(f) 
    return file

import os
import re
import pickle
import pandas as pd
import numpy as np
import warnings
import pickle
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim
import gensim.corpora as corpora
from gensim.matutils import Sparse2Corpus
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)

def load_pickle(name, query):
    '''
    Loads a pickle file given a query, which matches the
    file structure.
    ____
    Examples:
    ____
    models = load_pickle("H2_submissions_b0.1_a0.01", query = "model")
    corpus = load_pickle("H2_submissions_corpus", query = "corpus")
    df = load_pickle("H2_submissions", query = "data")
    '''
    if query == "corpus":
        with open(f"../data/modeling/{name}.pkl", "rb") as f: 
            corpus = pickle.load(f)
        return corpus
    if query == "model":
        with open(f"../data/models/{name}.pkl", "rb") as f: 
            model = pickle.load(f)
        return model
    if query == "data":
        with open(f"../data/clean/{name}.pkl", "rb") as f: 
            df = pickle.load(f)
        return df
    if query == "topic_df":
        with open(f"../data/modeling/topic_dfs/{name}.pkl", "rb") as f: 
            df = pickle.load(f)
        return df

# Find the topic number with the highest 
def dominant_topic(ldamodel, corpus, document, save_name = ""):
    '''
    Creates a dataframe, which indicates the dominant topic
    of a given document. 
    ___
    Examples:
    ___
    df_dominant = dominant_topic(models[model_name], corpus, df["org_text"])
    '''
    # init dataframe
    topics_df = pd.DataFrame()

    # GET MAIN TOPIC IN EACH DOCUMENT
    # Get through the pages
    for num, doc in enumerate(tqdm(ldamodel[corpus])):
        # Count number of list into a list
        if sum(isinstance(i, list) for i in doc)>0:
            doc = doc[0]

        doc = sorted(doc, key= lambda x: (x[1]), reverse=True)
    
        for j, (topic_num, prop_topic) in enumerate(doc):
            if j == 0: # => dominant topic
                # Get list prob. * keywords from the topic
                pk = ldamodel.show_topic(topic_num)
                topic_keywords = ', '.join([word for word, prop in pk])
                # Add topic number, probability, keywords and original text to the dataframe
                topics_df = topics_df.append(pd.Series([int(topic_num), np.round(prop_topic, 4),
                                                    topic_keywords, document[num]]),
                                                    ignore_index=True)
            else:
                break
                
    # Add columns name
    topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contribution', 'Keywords', 'Text']

    if save_name:
        with open(f"../data/modeling/topic_dfs/topic_dfs{save_name}.pkl", "wb") as f: 
            pickle.dump(topics_df, f)

    return topics_df

def topic_threshold(df, topic, threshold):
    '''
    Creates a subset of the data, which is documents that 
    have a topic_perc_contribution over a set threshold.
    '''
    return df[(df["Dominant_Topic"] == topic) & (df["Topic_Perc_Contribution"] > threshold)].sort_values("Topic_Perc_Contribution", ascending = False)

def query_topic(data, sub_size, query, topic = False):
    '''
    Queries a dataframe and gives a subset of sub_size length.
    This query contains both the topic and a query string. 
    Alternatively, setting topic = False only queries the dataframe
    and returns the sorted dataframe.
    '''
    if topic:
        data = data[(data["Text"].str.contains(query)) & (data["Dominant_Topic"] == topic)]
    else:
        data = data[data["Text"].str.contains(query)]
    return data.sort_values("Topic_Perc_Contribution", ascending=False).head(sub_size)