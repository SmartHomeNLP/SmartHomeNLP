from concat_files import get_data
from clean_comment import *


comments = pd.read_csv('../data/preprocessed/comments_collected.csv')
submissions = pd.read_csv('../data/preprocessed/submissions_collected.csv')

## unique: 
submissions.drop_duplicates(keep = "first", inplace = True) 
comments.drop_duplicates(keep = "first", inplace = True)

df = get_data(submissions, comments, filename = "data.csv")

df = remove_html(df)
df = hyphenate(df)
stop_words = get_stop_words()
english_vocab = get_english_words()

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
df['clean_text'] = df.clean_text.apply(clean_comment)

df = drop_rows(df)
df.to_csv("../data/preprocessed/data_clean.csv")