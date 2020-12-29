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
from evaluation_functions import * 

## get dataframes 
# melt: nice for picking out labels & specific topics.
# sort: nice for plotting distribution of our groups. 
melt, sort = label_counts("H1_labels")

## frequency plot
## would be nice to have security before energy management. 
frequency_plot(sort, drop = ["misc", "random"]) 

## import models ##
thread = pickle_load("H1_thread_models_b0.1_b1", "models")
tree = pickle_load("H1_tree_models", "models")

## select specific models screened prior ##
thread = thread['a0.01_b0.1_k30']
tree = tree['a0.01_b0.1_k30']

##### withing thread, looking at the big topics ######
## get out list of topics that are in categories
light_thread = label_subset(melt, ['comfort and lighting'], "thread30")
control_thread = label_subset(melt, ['control and connectivity'], "thread30")
security_thread = label_subset(melt, ['security'], "thread30")

## plot those categories
plot_images(thread, light_thread, "comfort & lighting", 3)
plot_images(thread, control_thread, "control & connectivity", 3) #generally more company focused
plot_images(thread, security_thread, "security", 2)

##### across thread and tree - what is recurring? ######
light_tree = label_subset(melt, ['comfort and lighting'], "tree30")
control_tree = label_subset(melt, ['control and connectivity'], "tree30")

plot_images(tree, light_tree, "comfort & lighting", 3)
plot_images(tree, control_tree, "control & connectivity", 3)

####### page through plots to find interesting subsets #######
## some weird spacing, but less important
## for our purposes here.. 
## trying to find specific companies: 
companies_thread = create_image_list(thread)

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

####### T-SNE #######
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/?fbclid=IwAR3xeAQMyimOPmtf9Jbk7agWqURyuWsvgMhIjVyh9ZnJtL20mA8vnefEVlk
# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors

# Get topic weights
topic_weights = []
for i, row_list in enumerate(tree[tree_corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x = tsne_lda[:,0], y = tsne_lda[:,1], color = mycolors[topic_num])
show(plot)