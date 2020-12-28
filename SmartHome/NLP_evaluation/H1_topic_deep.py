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

## check distribution acc. to our classification: 
with open("../data/evaluation/H1_labels.pkl", "rb") as f:
    labels = pickle.load(f) 

## to dataframe - melt - group
df = pd.DataFrame.from_dict(labels)
df_melt = pd.melt(df, value_vars = ['tree30', 'thread30'])
df_grouped = pd.DataFrame({'count' : df_melt.groupby(['value', 'variable']).size()}).reset_index()
df_sorted = df_grouped.sort_values(['count'], ascending = False).reset_index(drop=True)
df_sorted

## plot 
sns.barplot(data = df_sorted, x = "value", y = "count", hue = "variable")
plt.xticks(rotation=70)
plt.tight_layout()

## 

g.map_dataframe(sns.countplot, x = "topics" , y = "value")
g.set_axis_labels("topics", "coherence")
g.add_legend()

