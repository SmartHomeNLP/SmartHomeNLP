import pandas as pd 
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns 

with open(f"../data/evaluation/H1_thread_eval_b0.1_b1.pkl", "rb") as f:
        H2evaluation1 = pickle.load(f)

with open(f"../data/evaluation/H1_thread_eval_b0.01.pkl", "rb") as f:
        H2evaluation2 = pickle.load(f)

dots = sns.load_dataset("dots")
dots.head(2)
total = H2evaluation1.append(H2evaluation2)
total.dtypes
total

## seaborn 
# https://seaborn.pydata.org/examples/faceted_lineplot.html
sns.set_theme(style="ticks")
palette = sns.color_palette("rocket_r")
sns.relplot(
    data = total,
    x = "topics", y = "coherence",
    hue = "alpha", size="alpha", col = "eta",
    kind = "line", size_order=["0.01", "0.1", "1.0"],
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)


total = H2evaluation1.append(H2evaluation2)

H2plot = total.pivot(index='topics', columns='alpha', values='coherence')
H2plot.plot()

## need to run evaluation on the new shit..