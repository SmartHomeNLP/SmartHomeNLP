'''
proof that the old gathered stuff is exactly the 
same as we have when we download it again
(i.e. we don't need to re-download)
'''

import pandas as pd 
import os 
import datetime
#import dplython as tv 

os.getcwd()

### read just downloaded batch 
df_new = pd.read_csv("submissions2018-11.csv")

### read her stuff (raw) 
df_raw = pd.read_csv("../../DataSource_backup/reddit_submissions.csv")

## take out the dates that correspond: 
df_new['created_utc'].min() ## første november (check)
df_new['created_utc'].max() ## sidste november (check)


## conversions ##
## 1. convert our UNIX to timedate
df_new['time'] = pd.to_datetime(df_new['created_utc'], unit='s')
df_new = df_new.drop('created_utc', axis = 1)
## 2. convert her string to timedate
df_raw['time'] = pd.to_datetime(df_raw['created_utc'], format='%Y-%m-%d %H:%M:%S')
df_raw = df_raw.drop(['created_utc', 'link_id'], axis = 1)

df_new.columns == df_raw.columns

## subsetting ## 
## 1. selecting only the times between what we have in ours. 
df_new['time'].min() ## første november (check)
df_new['time'].max() ## sidste november (check)

## can this be done more tidy?
df_raw_sub = df_raw[(df_raw['time'] >= df_new['time'].min()) & (df_raw['time'] <= df_new['time'].max())]

## now only two observations differ: 
len(df_raw_sub)
len(df_new)

## what are those differences?
## none. they are the same..
differences = pd.concat([df_raw_sub, df_new]).drop_duplicates(keep=False)

