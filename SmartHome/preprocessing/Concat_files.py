'''
18-11-2020.
Checking how long this takes. 
'''
import re
import warnings
from contextlib import contextmanager
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os 
import glob
import time

## import files 
## perhaps specify dtypes: https://www.roelpeters.be/solved-dtypewarning-columns-have-mixed-types-specify-dtype-option-on-import-or-set-low-memory-in-pandas/
#comments = pd.read_csv('../data/preprocessed/comments_collected.csv')
#submissions = pd.read_csv('../data/preprocessed/submissions_collected.csv')

## unique: 
#submissions.drop_duplicates(keep = "first", inplace = True) 
#comments.drop_duplicates(keep = "first", inplace = True)

## columns 

#t1 = time.time()

def get_data(submissions, comments, filename = ""):
    df = pd.DataFrame()
    text_array = []
    link_id_array = []

    for link_id in submissions.loc[:, "link_id"].unique():
        org_text = str(submissions[submissions["link_id"] == link_id].values[0][2]) + " "
        if str(submissions[submissions["link_id"] == link_id].values[0][3]) != "nan":
            org_text += str(submissions[submissions["link_id"] == link_id].values[0][3]) + " "
        subset = comments[comments["link_id"] == link_id]
        text = org_text + " " + " ".join(subset["body"])
        text_array.append(text)
        link_id_array.append(link_id)
    df["text"] = text_array 
    df["link_id"] = link_id_array
    if filename:
        df.to_csv("../data/preprocessed/" + filename, index = False)
    return df

#test = get_data(submissions, comments, filename = "data.csv")

#print(time.time() - t1)


