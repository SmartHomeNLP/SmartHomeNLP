## INSPECT THE DATA
'''
Variables:
_____
link_id = the id of the submission that 
the comments are posted to
_____
parent_id = the id of the comment or submission 
that the comment is a "child" of. 
If the link id and parent id is the same, then the comment 
refers to the original post. 
_____
Data:
_____
It seems like comments and comments raw from the original are nearly identical. 
Need to figure out what is happening. 

'''

import pandas as pd
import glob

# MAIN:
import os
import datetime

path = glob.glob("../DataSource_backup/comments2019*.csv") #get the files from data

original_comments = pd.read_csv("../DataSource_backup/reddit_comments.csv")

original_comments["score"] = original_comments["score"].apply(lambda x: int(x)) #make integers

def read_files(file_list):
    '''
    Generator function for applying pandas "read_csv" to all 
    files in the file_list (here "path")
    '''
    for i in file_list:
        yield pd.read_csv(i)

combined_data = pd.concat(read_files(path), ignore_index=True) #concatenate dataframe and ignore index from previous dataframes

combined_data["created_utc"] = combined_data["created_utc"].apply(lambda x: datetime.datetime.fromtimestamp(x)) #get dates instead of timestamp

combined_data = pd.concat((original_comments, combined_data), ignore_index = True)

combined_data.to_csv("../data/preprocessed/comments_collected.csv", index = False)