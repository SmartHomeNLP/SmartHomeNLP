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

## coherence plots for evaluation
def eval_seaborn(df, name): 
    g = sns.FacetGrid(df, col = "beta", row = "variable", hue = "alpha")
    g.map_dataframe(sns.lineplot, x = "topics" , y = "value")
    g.set_axis_labels("topics", "evaluation value")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(name)
    g.add_legend()
    return(g)