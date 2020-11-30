import pandas as pd 
import os 
import re

### submissions
submissions = pd.read_csv("../data/preprocessed/submissions_nobots.csv")
submissions.columns

submissions = submissions.dropna()

### combine title and selftext 
submissions['text'] = submissions['title'] + '<bingo>' + submissions['selftext']
submissions['text'] = submissions['text'].apply(lambda x: x.lower())

## do these words actually occur (in submissions)
liste = ['security', 'trust', 'privacy']

for i in liste: 
    mentions = submissions[submissions['text'].str.contains(i)]
    print(f"{i} mentioned {len(mentions)} times")

mentions['text'].values

### comments 
comments = pd.read_csv("../data/preprocessed/comments_nobots.csv")

comments = comments.dropna()
comments['body'] = comments['body'].apply(lambda x: x.lower())

liste = ['security', 'privacy', 'trust']

for i in liste: 
    mentions = comments[comments['body'].str.contains(i)]
    print(f"{i} mentioned {len(mentions)} times")

mentions['body'].values