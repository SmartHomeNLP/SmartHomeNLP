
import pandas as pd 
import os 

os.getcwd()

## read the shit: 
df = pd.read_csv("submissions2019-08.csv")

df.columns

## how many authors. 
df['author'].value_counts()
df['author'].value_counts().mean()

## spread among the subreddits.
df['subreddit'].value_counts()

## how many comments are typical? (spoiler: few)
df['num_comments'].value_counts()

## when were they created..?
df['created_utc'].value_counts()