'''
23-11-2020
cleaning bots based on several approaches: 
1. regex 
2. posting very similar things 
3. manual
Finding the names of bots and filtering them out. 
'''

# run in shell followed by tables name: submissions or comments
# manually spot potential bots > return a list

import numpy as np
import pandas as pd
from sys import argv
import re
import pickle
import os 
import difflib

def diff_ratio(_a, _b):
    return difflib.SequenceMatcher(a=_a,b=_b).ratio()

### function used to clean comments 
def calc_stats(authors): 

    avg_diff_list = []

    num = 0
    for i in authors: 

        num += 1
        if(num % 1000 == 0): print(num)

        subset = comments[comments['author'] == i]

        if len(subset) > 1: 
            tweet1 = subset.sample(1)
            rest = pd.concat([subset, tweet1]).drop_duplicates(keep=False)
            tweet1 = tweet1.iloc[0]
            diff = rest['body'].str.slice(stop=200).fillna('').apply(lambda x: diff_ratio(tweet1['body'], x))
            avg_diff = diff.mean()
        
        else: 
            avg_diff = 0
            
        avg_diff_list.append(avg_diff)

    return avg_diff_list

## main function# 
def bot_clean(comments, submissions): 

    ## add a column & remove some 
    comments = comments[['body', 'author', 'created_utc']]

    ## finding problematic accounts
    problematic_accounts = []

    ## 445 comments found with "I am a bot" or variations.
    for i, txt in enumerate(comments.loc[:, 'body']): 
        bot = re.findall("(I|i).{0,3}('m|am).{0,3}(a).{0,3}(bot)", f"{txt}")
        if len(bot) != 0: 
            problematic_accounts.append(comments.loc[i, 'author'])

    ## contains duplicates, we remove these later.
    problematic_accounts

    ### trying the other system: self-similarity ### 

    # first we are just trying with a stupid approach # 
    author_list = list(comments['author'].unique())

    #running the calculate stats function
    avg_diff_list = calc_stats(author_list)

    #find the problematic accounts (avg_diff > 0.4) somewhat arbitrary.. 
    data = pd.DataFrame(list(zip(author_list, avg_diff_list)), columns =['author', 'avg_diff_list']) 
    problematic_accounts2 = list(data[data['avg_diff_list'] > 0.4]["author"])

    # merge the lists and keep [deleted] 
    problematic_accounts_comb = list(set(problematic_accounts + problematic_accounts))
    problematic_accounts_comb.remove('[deleted]')

    # use the manually screened bots (from later in the process). 
    # if you want to do this again go to *** MANUAL ***
    # here we just load the data 
    with open("manually_screened_bots.txt", "rb") as fp:   # Unpickling
        manual_bots = pickle.load(fp)

    all_bots = list(set(problematic_accounts_comb + manual_bots))

    ## remove bots from the original comments data & submissions: 
    comments = pd.read_csv("../data/preprocessed/comments_collected.csv")
    submissions = pd.read_csv("../data/preprocessed/submissions_collected.csv")

    clean_comments = comments[~comments['author'].isin(all_bots)]
    clean_submissions = submissions[~submissions['author'].isin(all_bots)]

    ## write the data 
    clean_comments.to_csv("../data/preprocessed/comments_nobots.csv", index = False)
    clean_submissions.to_csv("../data/preprocessed/submissions_nobots.csv", index = False)

## use the function: 
comments = pd.read_csv("../data/preprocessed/comments_collected.csv")
submissions = pd.read_csv("../data/preprocessed/submissions_collected.csv")

bot_clean(comments, submissions)

'''
### *** MANUAL *** ###
possible_bots = np.sort(clean_data.loc[clean_data.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

## manual screen: 
bots = []
for i in possible_bots:
    print(i)
    print(list(clean_data.loc[clean_data.author == i, "body"][:3]))
    print("="*50)
    while True:
        try: 
            status = input("Is this a bot? yes/no ")
            if status not in ["yes", "no"]:
                raise ValueError
            break
        except ValueError:
            print("Invalid. Type 'yes' or 'no'")
    if status == "yes":
        bots.append(i)

    print(bots)

# save the list
with open("manually_screened_bots.txt", "wb") as fp: 
    pickle.dump(bots, fp)
'''
