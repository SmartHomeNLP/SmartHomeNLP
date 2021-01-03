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
from visualization_functions import * 

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
misc_tree = label_subset(melt, ['misc'], 'tree30')
misc_thread = label_subset(melt, ['misc'], 'thread30')

plot_images(thread, misc_thread, "RQ1: Misc (thread)", 3)
plot_images(tree, misc_tree, "RQ1: Misc (tree)", 3)

## 
control_tree = label_subset(melt, ['control and connectivity'], 'tree30')
control_thread = label_subset(melt, ['control and connectivity'], 'thread30')