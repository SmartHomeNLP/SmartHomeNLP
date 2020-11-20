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
os.getcwd()
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\data\preprocessed' # use your path

## perhaps specify dtypes: https://www.roelpeters.be/solved-dtypewarning-columns-have-mixed-types-specify-dtype-option-on-import-or-set-low-memory-in-pandas/
comments = pd.read_csv(f'{path}/comments_collected.csv')
submissions = pd.read_csv(f'{path}/submissions_collected.csv')

## unique: 
submissions.drop_duplicates(keep = "first", inplace = True) 
comments.drop_duplicates(keep = "first", inplace = True)

## columns 
comments.columns 
submissions.columns
## downsample for now: 
#comments = comments[comments['created_utc'] > '2019-12-01 00:00:00']
#submissions = submissions[(submissions['created_utc'] >= '2019-12-01 00:00:00') & (submissions['created_utc'] <= comments_sub['created_utc'].max())]

## set-up
df_tree = pd.DataFrame(columns = ['tree_ids', 'tree_bodies'])

## only take comments that refer to a submission that we have. 
## i.e. filtering out the comments that refer back to things that
## were submitted before our first data.
df = comments[(comments['link_id'] == comments['parent_id']) & (comments.link_id.isin(submissions.link_id))]

## remove pretext in id. 
comments['id_parent_copy'] = [re.sub('.+_', '', x) for x in comments['parent_id']]

## big funky loop
start = time.time()
try:
    for first_com in df.loc[:, 'link_id'].unique(): # Comments from 7283 submission
        # For each parent_id get all comments with the same link_id
        # Get all comments within the same submission
        first_tier = comments[comments['link_id'] == first_com]
        # Initiate an empty tree
        tree_ids = []
        tree_bodies = []
        # Isolete the selected first tier comment >> could be more than one
        for init_i in list(df['id'][df['link_id'] == first_com]):
            tree_ids.append(init_i)
            tree_bodies.append(''.join(list(df['body'][df['id'] == init_i])))

            # concatenate all the children 
            i = []
            i.append(init_i)
            while not comments[comments['id_parent_copy'].isin(i)].empty:
                # all rows in sorted tmp inserted in tree_ids and tree bodies
                # all comments in the same tier are concatenate to each other
                sorted_tmp = comments[comments['id_parent_copy'].isin(i)].sort_values(by = ['created_utc'], ascending=False)
                num = list(df['id'][df['link_id'] == first_com]).index(init_i)
                tree_ids[num] += ' <NEW TIER> ' + ' <SAME TIER> '.join(list(sorted_tmp['id']))
                tree_bodies[num] += ' <NEW TIER> ' + ' <SAME TIER> '.join([str(elm) for elm in list(sorted_tmp['body'])])
                i = list(sorted_tmp['id'])

        # store in a new database
        for n_row in range(len(tree_ids)):
            df_tree = df_tree.append({'tree_ids': tree_ids[n_row], 'tree_bodies': tree_bodies[n_row]}, ignore_index=True)
except:
    first_com
end = time.time() 
print(f'time elapsed = {end - start}')

df_tree['id'] = [re.sub('\\s.*', '',x) for x in  df_tree['tree_ids']]
df_tree = df_tree.merge(comments.loc[:, ['id','link_id']], on = 'id')
df_tree = df_tree.merge(submissions.loc[:, ['link_id', 'title', 'selftext']], on='link_id')

path = 'c:\\Users\\95\\Dropbox\\MastersSem1\\NLP\\SmartHome\\SmartHomeNLP\\SmartHome\\data\\preprocessed'
df_tree.to_csv(f'{path}\df_tree.csv', index=False)

