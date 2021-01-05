'''
This document finds the topic prevalence given a (1) dataframe
containing the dominant topic per document and a list of 
labels for the documents. 
'''

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
import matplotlib.pyplot as plt 
from doc_from_topics import *

### Loading the labels from evaluation folder

with open(f"../data/evaluation/H1_labels.pkl", "rb") as f: 
            labels = pickle.load(f)

with open(f"../data/evaluation/H2_labels.pkl", "rb") as f: 
            labels_sub = pickle.load(f)

labels_sub = labels_sub["sub30_tags"]
### Finding the relevant item from the dictionary
labels_tree = labels["tree30"]
labels_thread = labels["thread30"]

### Loading the dataframe of dominant topics
tree_df = load_pickle("topic_dfsH1_tree_30", "topic_df")
thread_df = load_pickle("topic_dfsH1_thread_30", "topic_df")
sub_df = load_pickle("topic_dfsH2_submissions_30", "topic_df")

def get_topic_label(df, topic_labels):
    '''
    Given a dataframe and a list of topic labels,
    this function adds a new column to the dataframe,
    containing the topic labels corresponding to each
    topic. 
    Assumes that the topic_labels are ordered according
    to their topic number i.e. topic_labels[i] is the label
    for topic 0.
    Returns the dataframe with the added column.
    '''
    topic_set = set(topic_labels)
    index_labels = {j: [i for i, ele in enumerate(topic_labels) if ele == j] for j in topic_set}
    dominant_topics = df["Dominant_Topic"].values
    label_col = [j for i in dominant_topics for j in index_labels if i in index_labels[j]]
    df["Topic_Label"] = label_col
    return df

### Run the function
tree_df = get_topic_label(tree_df, labels_tree)
thread_df = get_topic_label(thread_df, labels_thread)
sub_df = get_topic_label(sub_df, labels_sub)

### Plot the relative contribution of each labelled topic
fig, ax = plt.subplots(3, sharex=True, figsize=(7, 7))
labels_plot = ["Tree", "Thread", "Submissions"]
colors_plot = ["teal", "purple", "green"]
for i, ele in enumerate([tree_df, thread_df, sub_df]):
    counts = ele["Topic_Label"].value_counts(normalize = True)
    ax[i].barh(counts.keys(), counts, color = colors_plot[i])
    ax[i].set_title(labels_plot[i])
fig.tight_layout()
fig.savefig("../Figure/Topic_Prevalence.png")