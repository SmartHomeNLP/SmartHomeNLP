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


## get all the files: 
## we ran it in several batches: 
with open("../data/evaluation/evaluation_metrics.pkl", "rb") as f:
        eval1 = pickle.load(f) 

with open("../data/evaluation/H2_evaluation_metrics.pkl", "rb") as f:
        eval2 = pickle.load(f)

with open("../data/evaluation/H1_tree_eval_b1.0.pkl", "rb") as f:
        eval3 = pickle.load(f)

with open("../data/evaluation/H1_tree_metrics_b0.01_b0.1.pkl", "rb") as f: 
        eval4 = pickle.load(f)

## combine the two: 
eval = pd.concat([eval1, eval2, eval3, eval4])

## correct data type for topics, eta, alpha: 
eval[['topics', 'alpha', 'eta']] = eval[['topics', 'alpha', 'eta']].apply(pd.to_numeric) 

## arun between 0-1: 
eval = eval.apply(lambda x: ((x-min(x))/(max(x)-min(x))) if x.name == 'arun' else x)

## long format for plotting: 
eval_melt = pd.melt(eval, id_vars = ['hypothesis', 'condition', 'topics', 'alpha', 'eta'], value_vars = ['coherence_cv', 'cao_juan', 'arun'])

## 1. H1_tread: 5-50, all combinations: 
thread_melt = eval_melt[(eval_melt.hypothesis == "H1") & (eval_melt.condition == "thread") & (eval_melt.topics <= 50)] # & (eval_melt.topics <= 50)
thread_melt

# plotting
g = sns.FacetGrid(thread_melt, col = "eta", row = "variable", hue = "alpha")
g.map_dataframe(sns.lineplot, x = "topics" , y = "value")
g.set_axis_labels("topics", "coherence")
g.set_title("just one please")
g.fig.suptitle('THIS IS A TITLE, YOU BET')
g.add_legend()

# save figure: 
g.savefig("../Figure/H1_thread.png")

# 2. H1_thread vs. tree: 
# make a call (based on this AND interpretability).
H1_comparison = eval_melt[(eval_melt.hypothesis == "H1") & (eval_melt.alpha == 0.01) & (eval_melt.eta == 0.10)]
g = sns.FacetGrid(H1_comparison, col = "variable", hue = "condition", sharey = False)
g.map_dataframe(sns.lineplot, x = "topics" , y = "value")
g.set_axis_labels("topics", "coherence")
g.add_legend()

# save figure
g.savefig("../Figure/H1_comparison.png")

# 3. H2_submissions for 5-100 (only b01/a001): 
H2_sub = eval_melt[(eval_melt.hypothesis == "H2") & (eval_melt.alpha == 0.01) & (eval_melt.eta == 0.10)]
plot = sns.lineplot(data = H2_sub, x = "topics", y = "value", hue = "variable")

# save figure: 
plot.figure.savefig("../Figure/H2_submissions.png")

# 4. H2_submissions (5-50)
H2_sub2 = eval_melt[(eval_melt.hypothesis == "H2") & (eval_melt.topics <= 50)]
g = sns.FacetGrid(H2_sub2, col = "eta", row = "variable", hue = "alpha")
g.map_dataframe(sns.lineplot, x = "topics" , y = "value")
g.set_axis_labels("topics", "coherence")
g.add_legend()

g.savefig("../Figure/H2_grid.png")