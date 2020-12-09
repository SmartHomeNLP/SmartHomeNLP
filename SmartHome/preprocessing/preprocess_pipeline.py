#from concat_files import *
from clean_comment import *
import pandas as pd
import timeit

global_start = timeit.default_timer()
#comments = pd.read_csv('../data/preprocessed/comments_nobots.csv')
#submissions = pd.read_csv('../data/preprocessed/submissions_nobots.csv')

## unique: 
#submissions.drop_duplicates(keep = "first", inplace = True) 
#comments.drop_duplicates(keep = "first", inplace = True)

start = timeit.default_timer()
print("starting to concatenate submissions and comments")
### CONCAT SUBMISSIONS AND COMMENTS:
df = pd.read_csv("../data/preprocessed/data_all.csv")
df.drop_duplicates(keep = "first", inplace = True)
#df = get_data(submissions, comments, filename = "data.csv")
#df.drop('link_id', inplace = True, axis = 1)
stop = timeit.default_timer()
print(f"it took: {(stop - start)/60} minutes to concatenate submissions and comments")

start = timeit.default_timer()
print("doing smaller cleaning tasks")
### REMOVE HTML 
df = remove_html(df)
df.drop('text', inplace = True, axis = 1)

### CONCAT WORDS WITH HYPHENS IN BETWEEN
df = hyphenate(df)

### GET STOP WORDS AND VOCAB
stop = timeit.default_timer() 
print(f"it took: {(stop - start)/60} minutes to do cleaning tasks")

## APPLY FUNCTION TO COMMENT
start = timeit.default_timer()
print("starting clean the threads")
df = clean_comment(df, "clean_text")
stop = timeit.default_timer() 
print(f"it took {(stop-start)/60} minutes to clean the threads")

df = drop_rows(df)

df.to_csv("../data/preprocessed/data_new_clean.csv", index = False)
global_end = timeit.default_timer()
print(f"it took {(global_end - global_start)/60} minutes to run the whole damn thing")
