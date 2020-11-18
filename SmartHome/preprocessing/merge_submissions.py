'''
VMP 17-11-2020:
Reading all the new raw submission files from the data/raw
folder. (submissions2019-07.csv - submissions2020-04.csv). 
Making the format compatible with the format in the 
previously used data (DataSource_backup/reddit_submissions.csv). 
Then writing the combined csv file to data/preprocessed as 
"submissions_collected_raw.csv".
'''

import pandas as pd 
import os 
import datetime
import glob

os.getcwd()

### path to the new raw files (after decompression)
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\data\raw' # use your path
all_files = glob.glob(path + "/submissions*.csv") #make sure only to load the submissions. 

### read all of the new submissions
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

new_frame = pd.concat(li, axis=0, ignore_index=True)

### read all of the old submissions 
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\DataSource_backup'
old_frame = pd.read_csv(f'{path}/reddit_submissions.csv')

### making sure that we have consistent format 
new_frame.columns #does not have link_id
old_frame.columns #does have link_id 

### creating link_id by inserting t3_ to the beginning of id. 
new_frame = new_frame.assign(link_id = lambda x: "t3_" + x.id)
new_frame.head()

### making the time-format compatible and series. ### 
## 1. convert new data to UNIX to timedate
new_frame['created_utc'] = pd.to_datetime(new_frame['created_utc'], unit='s')
type(new_frame['created_utc'])
## 2. convert old data to timedate
old_frame['created_utc'] = pd.to_datetime(old_frame['created_utc'], format='%Y-%m-%d %H:%M:%S')
type(old_frame['created_utc'])

## checking whether the formats are now identical. 
new_frame.columns == old_frame.columns
new_frame.dtypes == new_frame.dtypes

## binding it together 
collected = pd.concat([old_frame, new_frame], ignore_index=True)

len(old_frame)

## write csv 
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\data\preprocessed'
collected.to_csv(f'{path}\submissions_collected_raw.csv', index=False)

## checking whether it worked properly
collected_test = pd.read_csv(f'{path}\submissions_collected_raw.csv')
collected_test.columns 
collected_test.dtypes 