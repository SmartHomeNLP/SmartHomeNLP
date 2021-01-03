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

from evaluation_functions import *

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

## correct data type for topics, eta, alpha and rename: 
eval[['topics', 'alpha', 'eta']] = eval[['topics', 'alpha', 'eta']].apply(pd.to_numeric) 
eval = eval.rename(columns = {'eta': 'beta', 'coherence': 'evaluation'}, inplace = False)

## arun between 0-1: 
eval = eval.apply(lambda x: ((x-min(x))/(max(x)-min(x))) if x.name == 'arun' else x)

## long format for plotting: 
eval_melt = pd.melt(eval, id_vars = ['hypothesis', 'condition', 'topics', 'alpha', 'beta'], value_vars = ['coherence_cv', 'cao_juan', 'arun'])

## subsets for H1_thread, H1_tree, H2_submissions
thread_melt = eval_melt[(eval_melt.hypothesis == "H1") & (eval_melt.condition == "thread") & (eval_melt.topics <= 50)] 
tree_melt = eval_melt[(eval_melt.hypothesis == "H1") & (eval_melt.condition == "tree") & (eval_melt.topics <= 50)] 
sub_melt = eval_melt[(eval_melt.hypothesis == "H2") & (eval_melt.topics <= 50)]

# plotting
thread_plot = eval_seaborn(thread_melt, "RQ1: Thread Evaluation")
tree_plot = eval_seaborn(tree_melt, "RQ1: Tree Evaluation")
sub_plot = eval_seaborn(sub_melt, "RQ2: Submissions Evaluation")

# save figures: 
thread_plot.savefig("../Figure/H1_thread.png")
tree_plot.savefig("../Figure/H1_tree.png")
sub_plot.savefig("../Figure/H2_submissions.png")

