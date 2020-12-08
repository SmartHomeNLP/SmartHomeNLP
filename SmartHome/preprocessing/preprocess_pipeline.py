from concat_files import *
from clean_comment import *
import pandas as pd
import timeit

global_start = timeit.default_timer()
#comments = pd.read_csv('../data/preprocessed/comments_nobots.csv')
#submissions = pd.read_csv('../data/preprocessed/submissions_nobots.csv')

## unique: 
submissions.drop_duplicates(keep = "first", inplace = True) 
comments.drop_duplicates(keep = "first", inplace = True)

start = timeit.default_timer()
print("starting to concatenate submissions and comments")
### CONCAT SUBMISSIONS AND COMMENTS:
df = pd.read_csv("../data/preprocessed/data_all.csv")
df = get_data(submissions, comments, filename = "data.csv")
df.drop('link_id', inplace = True, axis = 1)
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

### NOTE:
## Does not remove digits or *many* of the underscores. These needs to removed entirely.

## EATS UP THE WHOLE COMPUTER, BE CAREFUL!

# import multiprocessing

# start_time = timeit.timeit()
# processes = []

# for comment in df["clean_text"].values:
#     p = multiprocessing.Process(target = clean_comment, args = (comment,))
#     processes.append(p)
#     p.start()

# for process in processes:
#     processes.join()

# Apply function to clean the comment - maybe not needed.
start = timeit.default_timer()
print("starting clean the threads")
num = 0
df['clean_text'] = df.clean_text.apply(clean_comment)
stop = timeit.default_timer() 
print(f"it took {(stop-start)/60} minutes to clean the threads")

df = drop_rows(df)

df.to_csv("../data/preprocessed/data_new_clean.csv", index = False)
global_end = timeit.default_timer()
print(f"it took {(global_end - global_start)/60} minutes to run the whole damn thing")

df["text"].values