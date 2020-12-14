import pandas as pd 
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle

## 
with open("../data/evaluation/eval_metrics.pkl", "rb") as f:
        H2evaluation = pickle.load(f)

## plot: could be made prettier. 
## looks good around ~20-30 topics. 
## need additional metrics.
## all eta = 0.01
H2plot = H2evaluation.pivot(index='topics', columns='alpha', values='coherence')
H2plot.plot()

