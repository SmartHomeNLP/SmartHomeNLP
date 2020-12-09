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
import timeit

#test = pd.read_csv("../data/preprocessed/data_all.csv")

## import files 
## perhaps specify dtypes: https://www.roelpeters.be/solved-dtypewarning-columns-have-mixed-types-specify-dtype-option-on-import-or-set-low-memory-in-pandas/
comments = pd.read_csv('../data/preprocessed/comments_nobots.csv')
submissions = pd.read_csv('../data/preprocessed/submissions_nobots.csv')



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

def append_data(dictionary, text, link_id, thread, tree, is_sub):
    dictionary["text_array"].append(text)
    dictionary["link_id_array"].append(link_id)
    dictionary["thread_array"].append(thread)
    dictionary["tree_array"].append(tree)
    dictionary["is_sub_array"].append(is_sub)

def get_data_extended(submissions, comments, filename = ""):
    comments['id_parent_copy'] = [re.sub('.+_', '', x) for x in comments['parent_id']]
    df = pd.DataFrame()
    text_array = []
    thread_array = []
    tree_array = []
    is_sub_array = []
    link_id_array = []
    dictionary = {"text_array": text_array, "thread_array": thread_array,"link_id_array": link_id_array, "tree_array": tree_array, "is_sub_array": is_sub_array}

    for thread, link_id in enumerate(submissions.loc[:, "link_id"].unique()):
        org_text = str(submissions[submissions["link_id"] == link_id].values[0][2]) + " "
        if str(submissions[submissions["link_id"] == link_id].values[0][3]) != "nan":
            org_text += str(submissions[submissions["link_id"] == link_id].values[0][3])
        
        append_data(dictionary, org_text, link_id, thread, 0, 1)

        subset = comments[comments["link_id"] == link_id]
        top_comments = subset[subset["parent_id"] == link_id]

        for tree, top_com in enumerate(top_comments.loc[:, "id"]):
            text = top_comments[top_comments["id"] == top_com]["body"].values[0]
            append_data(dictionary, text, link_id, thread, tree+1, 0)
            parents = [top_com]
            while not subset[subset["id_parent_copy"] == parents[0]].empty:
                
                for child in subset[subset["id_parent_copy"] == parents[0]].loc[:, "id"]:
                    append_data(dictionary, subset[subset["id"] == child]["body"].values[0], link_id, thread, tree+1, 0)
                    parents.append(child)
                parents.pop(0)
                

    df["text"], df["link_id"], df["thread"], df["tree"], df["sub"] = dictionary["text_array"], dictionary["link_id_array"], dictionary["thread_array"], dictionary["tree_array"], dictionary["is_sub_array"]

    if filename:
        df.to_csv("../data/preprocessed/" + filename, index = False)
    return df

start = timeit.default_timer()

test = get_data_extended(submissions, comments, "data_all.csv")

stop = timeit.default_timer()

print(f"It took {stop - start} to finish the function")
