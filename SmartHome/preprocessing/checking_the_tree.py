'''
17-11-2020
Just checking the structure of df_tree.csv.
This should be deleted at some point.
'''

import pandas as pd 

## reading the file 
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\DataSource_backup'
df_tree = pd.read_csv(f'{path}\df_tree.csv')

df_tree.head()
df_tree.group

df_tree.groupby('group1')['link_id'].agg({'mean_col' : np.mean()})

df_tree["link_id"].describe()

## making a subset 
test = df_tree[df_tree['link_id'] == 't3_91ad8v']
test.iloc[0]['tree_ids']
test.head()
## same tier. 
df_tree.head()


##### loading test & train 
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\DataSource_backup'
df_sub = pd.read_csv(f'{path}\sub_onetree_test.csv')

## has removed 10% of data. 
df_sub.head(2)
df_sub.iloc[1]["clean_text"]