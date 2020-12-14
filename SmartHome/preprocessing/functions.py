import functools 
import timeit 
import numpy as np
from sys import argv
import re
import pickle
import os 
import difflib
import pandas as pd 
import datetime
from tqdm import tqdm
import warnings
from contextlib import contextmanager
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import html

def timer(func): 
    """Prints run-time""" 
    @functools.wraps(func)

    def wrapper_timer(*args, **kwargs): 
        
        ## start time
        start_time = timeit.default_timer()

        ## print that you start
        print(f"Starting {func,__name__!r}")

        ## generate the value from the function
        value = func(*args, **kwargs) 
        
        ## end time 
        end_time = timeit.default_timer() 

        ## calculate run-time
        run_time = (end_time - start_time)/60 

        ## print statement 
        print(f"Finished {func,__name__!r} in {run_time:.4f} minutes") 
        
        return value 
    return wrapper_timer


## MERGE SUBMISSIONS ### 

@timer
def merge_submissions():
    
    filenames = []
    for file in os.listdir("../data/raw"):
        if file.startswith("submissions"):
            filenames.append(file)

    ### read all of the new submissions
    new_sub = []

    for filename in filenames:
        df = pd.read_csv(f"../data/raw/{filename}", index_col=None, header=0)
        new_sub.append(df)

    new_frame = pd.concat(new_sub, axis=0, ignore_index=True)

    ### read all of the old submissions 
    old_frame = pd.read_csv('../DataSource_backup/reddit_submissions.csv')

    ### creating link_id by inserting t3_ to the beginning of id. 
    new_frame = new_frame.assign(link_id = lambda x: "t3_" + x.id)

    ### making the time-format compatible and series. ### 

    ## 1. convert new data to UNIX to timedate
    new_frame['created_utc'] = pd.to_datetime(new_frame['created_utc'], unit='s')

    ## 2. convert old data to timedate
    old_frame['created_utc'] = pd.to_datetime(old_frame['created_utc'], format='%Y-%m-%d %H:%M:%S')

    ## binding it together 
    collected = pd.concat([old_frame, new_frame], ignore_index=True)

    ## write csv 
    collected.drop_duplicates(keep = "first", inplace = True) 

    with open("../data/preprocessed/submissions_collected.pkl", 'wb') as sub:
        pickle.dump(collected, sub)
    
## MERGE COMMENTS ## 
@timer
def merge_comments():

    filenames = []
    for file in os.listdir("../data/raw"):
        if file.startswith("comments"):
            filenames.append(file)

    new_comment = []

    for filename in filenames:
        df = pd.read_csv(f"../data/raw/{filename}", index_col=None, header=0)
        new_comment.append(df)

    
    original_comments = pd.read_csv("../DataSource_backup/reddit_comments.csv")

    # as integeres: 
    original_comments["score"] = original_comments["score"].apply(lambda x: int(x))

    combined_data = pd.concat(new_comment, ignore_index=True) #concatenate dataframe and ignore index from previous dataframes

    combined_data["created_utc"] = combined_data["created_utc"].apply(lambda x: datetime.datetime.fromtimestamp(x)) #get dates instead of timestamp

    combined_data = pd.concat((original_comments, combined_data), ignore_index = True)

    combined_data.drop_duplicates(keep = "first", inplace = True) 

    with open("../data/preprocessed/comments_collected.pkl", 'wb') as com:
        pickle.dump(combined_data, com)

### BOTS ### 

def diff_ratio(_a, _b):
    return difflib.SequenceMatcher(a=_a,b=_b).ratio()

### function used to clean comments 
def calc_stats(authors, comments): 

    avg_diff_list = []

    for i in tqdm(authors): 

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
@timer
def bot_clean(manual = False, done = False): 

    if done: 
        print("skipped bot cleaning")
        return(None)

    with open("../data/preprocessed/comments_collected.pkl", 'rb') as com:
        comments = pickle.load(com)
    
    with open("../data/preprocessed/submissions_collected.pkl", 'rb') as sub:
        submissions = pickle.load(sub)

    ## finding problematic accounts
    problematic_accounts = []

    ## 1: REGEX
    for i, txt in enumerate(comments.loc[:, 'body']): 
        bot = re.findall("(I|i).{0,3}('m|am).{0,3}(a).{0,3}(bot)", f"{txt}")
        if len(bot) != 0: 
            problematic_accounts.append(comments.loc[i, 'author'])

    ## contains duplicates, we remove these later.
    problematic_accounts
    print(f"regex bots found: {len(problematic_accounts)}")
    ### 2: self-similarity ### 

    # first we are just trying with a stupid approach # 
    author_list = list(comments['author'].unique())

    #running the calculate stats function
    avg_diff_list = calc_stats(author_list, comments)

    #find the problematic accounts (avg_diff > 0.4) somewhat arbitrary.. 
    data = pd.DataFrame(list(zip(author_list, avg_diff_list)), columns =['author', 'avg_diff_list']) 
    problematic_accounts2 = list(data[data['avg_diff_list'] > 0.4]["author"])
    print(f"self-similarity bots found: {len(problematic_accounts2)}")

    # merge the lists and keep [deleted] 
    problematic_accounts_comb = list(set(problematic_accounts + problematic_accounts2))
    problematic_accounts_comb.remove('[deleted]')
    print(f"total bots (excluding deleted): {len(problematic_accounts_comb)}")

    # remove the automatically generated problematic accounts: 
    comments = comments[~comments['author'].isin(problematic_accounts_comb)]
    submissions = submissions[~submissions['author'].isin(problematic_accounts_comb)]

    if manual: 
        possible_bots = np.sort(comments.loc[comments.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())
        
        ## manual screen: 
        bots = []
        for i in possible_bots:
            print(f"name: {i}")
            print(list(comments.loc[comments.author == i, "body"][:3]))
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

        # save the list
        with open("manually_screened_bots.txt", "wb") as fp: 
            pickle.dump(bots, fp)

    # use the manually screened bots (from later in the process). 
    # if you want to do this again go to *** MANUAL ***
    # here we just load the data 
    with open("manually_screened_bots.txt", "rb") as fp:   # Unpickling
        manual_bots = pickle.load(fp)

    ## remove bots from the original comments data & submissions: 
    clean_comments = comments[~comments['author'].isin(manual_bots)]
    clean_submissions = submissions[~submissions['author'].isin(manual_bots)]

    ## write the data 
    with open("../data/preprocessed/comments_nobots.pkl", 'wb') as com:
        pickle.dump(clean_comments, com)
    
    with open("../data/preprocessed/submissions_nobots.pkl", 'wb') as sub:
        pickle.dump(clean_submissions, sub)

## CONCAT COMMENTS & SUBMISSIONS ##

# helper function
def append_data(dictionary, text, thread, tree, is_sub, op_comment):
    dictionary["text_array"].append(text)
    dictionary["thread_array"].append(thread)
    dictionary["tree_array"].append(tree)
    dictionary["is_sub_array"].append(is_sub)
    dictionary["op_comment_array"].append(op_comment)

# main function 
@timer
def concat_data(done = False):

    if done: 
        print("skipped concatenating files")
        return(None)

    ## load stuff
    import pickle
    with open("../data/preprocessed/comments_nobots.pkl", 'rb') as com:
        comments = pickle.load(com)
    comments
    with open("../data/preprocessed/submissions_nobots.pkl", 'rb') as sub:
        submissions = pickle.load(sub)

    comments['id_parent_copy'] = [re.sub('.+_', '', x) for x in comments['parent_id']]
    df = pd.DataFrame()
    text_array = []
    thread_array = []
    tree_array = []
    is_sub_array = []
    op_comment_array = []
    dictionary = {"text_array": text_array, "thread_array": thread_array, "tree_array": tree_array, "is_sub_array": is_sub_array, "op_comment_array": op_comment_array}

    for thread, link_id in enumerate(tqdm(submissions.loc[:, "link_id"].unique())):
        op_id = submissions[submissions["link_id"] == link_id]["author"].values[0]
        org_text = str(submissions[submissions["link_id"] == link_id].values[0][2]) + " "
        if str(submissions[submissions["link_id"] == link_id].values[0][3]) != "nan":
            org_text += str(submissions[submissions["link_id"] == link_id].values[0][3])
        
        append_data(dictionary, org_text, thread, 0, 1, 0)

        subset = comments[comments["link_id"] == link_id]
        top_comments = subset[subset["parent_id"] == link_id]

        for tree, top_com in enumerate(top_comments.loc[:, "id"]):
            text = top_comments[top_comments["id"] == top_com]["body"].values[0]
            if top_comments[top_comments["id"] == top_comments]["author"].values[0] == op_id:
                append_data(dictionary, text, thread, tree+1, 0, 1)
            else:
                append_data(dictionary, text, thread, tree+1, 0, 0)
            parents = [top_com]
            while not subset[subset["id_parent_copy"] == parents[0]].empty:
                
                for child in subset[subset["id_parent_copy"] == parents[0]].loc[:, "id"]:
                    if subset[(subset["id_parent_copy"] == parents[0]) & (subset["id"] == child)]["author"].values[0] == op_id:
                        append_data(dictionary, subset[subset["id"] == child]["body"].values[0], thread, tree+1, 0, 1)
                    else:
                        append_data(dictionary, subset[subset["id"] == child]["body"].values[0], thread, tree+1, 0, 0)
                    parents.append(child)
                parents.pop(0)
                
    df["text"], df["thread"], df["tree"], df["sub"], df["op_comment"] = dictionary["text_array"], dictionary["thread_array"], dictionary["tree_array"], dictionary["is_sub_array"], dictionary["op_comment_array"]

    with open("../data/preprocessed/concat_data.pkl", 'wb') as concat:
        pickle.dump(df, concat)

## SMALLER CLEANING TASKS ## 
@timer
def small_cleaning(done = False):

    if done: 
        print("skipping cleaning")
        return(None)

    with open("../data/preprocessed/concat_data.pkl", 'rb') as concat:
        df = pickle.load(concat)

    ## hyperlinks
    df["clean_text"] = [re.sub(r"((?:https?:\/\/)(?:\w[^\)\]\s]*))",'', x) for x in df['text']]
    df['clean_text'] =  [html.unescape(x) for x in df['clean_text']]

    ## hyphenate
    df['clean_text'] = [re.sub(r"\b(\w*)-(\w*)\b", r"\g<1>_\g<2>", x) for x in df['clean_text']]

    with open("../data/preprocessed/concat_data1.pkl", 'wb') as concat:
        pickle.dump(df, concat)

def get_stop_words():
    stop_words = stopwords.words('english')

    #Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
    more_stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
                "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
                "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
                "during", "each", "few", "for", "from", "further", "had", "has", "have",
                "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
                "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me",
                "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
                "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
                "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
                "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
                "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
                "we're", "we've", "were", "what", "what's", "when", "when's", "where",
                "where's", "which", "while", "who", "who's", "whom", "why", "why's",
                "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
                "yours", "yourself", "yourselves"]

    stop_words.extend(list(set(more_stopwords) - set(stop_words)) + ['etc', 'however', 'there', 'also', 'digit'])

    return stop_words

def get_english_words():
    nltk.download("words")
    return set(w.lower() for w in nltk.corpus.words.words())

def drop_rows(df):
    print("---- Dropping rows ----")
    print(f"Original no. rows: {len(df)}")
    df.drop(df.index[df.clean_text == "",].tolist(), axis=0, inplace=True)
    print(f"No. rows after dropping empty strings: {len(df)}")
    df.drop(df.index[df.clean_text.isna(),].tolist(), axis=0, inplace=True)
    # remove rows with less than 15 words (short observations)
    print(f"No. rows after dropping NAs: {len(df)}")
    
    #NOTE: Not removing comments with length < 15, as we are now taking individual comments and concatenating them
    # Consider doing this step after concat
    
    #df = df.loc[df['clean_text'].map(lambda x: len(str(x).strip().split())) > 15,]
    #print(f"No. rows after dropping strings < length 15: {len(df)}")
    return df

@timer
def clean_comment(done = False, del_tags = ['NUM', 'PRON', 'ADV', 'DET', 'AUX', 'SCONJ', 'PART']):
    
    if done: 
        print("skipping cleaning")
        return(None)

    with open("../data/preprocessed/concat_data1.pkl", 'rb') as concat:
        df = pickle.load(concat)

    stop_words = get_stop_words()
    english_vocab = get_english_words()
    
    comments = df['clean_text'].values #get values out as array
    data = []

    nlp = spacy.load("en_core_web_sm")

    for comment in tqdm(comments):
        #comment = re.sub(r"(<SUB>|nan|<NEW TIER>|<SAME TIER>)", "", comment) #deleting the markers and nan, but we don't have any of these.
        comment = comment.lower() # should be heavily considered in terms of event detection as we will want to detect capitalized letters as a feature
        comment = re.sub(r'&#x200B', ' ', comment) # character code for a zero-width space
        comment = re.sub(r'remindme![\w\s\W]*$', ' ', comment) # remove call to remind me bot
        comment = re.sub(r'\n', ' ', comment) # remove new line formatting
        comment = re.sub(r'(\[deleted\]|\[removed\])', '', comment)
        comment = re.sub(r"[^\w\s]", ' ', comment) # punctuation and emoji
        comment = re.sub(r'(\s_|_\s)', '', comment) # remove underscores around a words (italics)
        comment = re.sub(r"_", " ", comment) #replace underscores with space
        comment = re.sub(r"[\d]+", "", comment) #remove digits

        #print(f"Substitutions: {(timeit.timeit() - start)/60} minutes")
        #seems like this could be speeded up:
        # ______________________

        # detect no english comments and remove them 
        #nltk.download('words')
        text_vocab = set(w for w in comment.strip().split() if w.isalpha())
        unusual = text_vocab.difference(english_vocab) 

        # empty comments where 70% words not english, slangs, deleted
        try:
            if len(unusual)/len(text_vocab) > 0.7:
                comment = ''
        except ZeroDivisionError:
            pass
        
        #print(f"Language/slang detection: {(timeit.timeit() - start)/60} minutes")
        # remove stop_words
        comment_token_list = [word for word in comment.strip().split() if word not in stop_words and len(word)>1]


        comment_text = nlp(' '.join(comment_token_list))

        comment_token_list = [token.lemma_ for token in comment_text if token.pos_ not in del_tags]

        comment = ' '.join(comment_token_list)

        #print(f"Join strings: {(timeit.timeit() - start)/60} minutes")

        #NOTE digits within string

        data.append(comment)
        #progress_bar.update(1)
    
    # something
    df["clean_text"] = data
    df = drop_rows(df)
    
    # write
    with open("../data/preprocessed/concat_data2.pkl", 'wb') as cleaned:
        pickle.dump(df, cleaned)

## preprocessing for specific purposes: 
@timer
def H1_preprocess(get = [], done = False): 

    if done: 
        print("skipping specific preprocessing")
        return(None)
    
    ## goes for all the functions: 
    with open("../data/preprocessed/concat_data2.pkl", 'rb') as cleaned:
        data = pickle.load(cleaned)

    if "thread" in get: 
        res = pd.DataFrame()
        clean_texts = []
        org_texts = []
        for i in tqdm(set(data["thread"].values)):
            subset = data[data["thread"] == i]
            subset_clean = subset["clean_text"].values
            subset_org = subset["text"].values
            subset_clean = [str(x) for x in subset_clean] #getting an error because of a number????????? Why is this not removed?7
            subset_org = [str(x) for x in subset_org]
            text_clean = " ".join(subset_clean)
            text_org = " ".join(subset_org)
            clean_texts.append(text_clean)
            org_texts.append(text_org)
        res["clean_text"] = clean_texts
        res["org_text"] = org_texts
        with open("../data/clean/H1_thread.pkl", 'wb') as H1_thread:
            pickle.dump(res, H1_thread)

    if "tree" in get: 
        res = pd.DataFrame()
        clean_texts = []
        org_texts = []
        for i in tqdm(set(data["thread"].values)):
            submission_string = data[(data["thread"] == i) & (data["sub"] == 1)]
            if len(submission_string) != 0: #NOTE: some of the submissions have been dropped somewhere, so it has to be done this way
                submission_string_clean = submission_string["clean_text"].values[0]
                submission_string_org = submission_string["text"].values[0]
            else:
                submission_string_clean, submission_string_org = "", ""
            for j in set(data[data["thread"] == i]["tree"].values):
                subset = data[(data["thread"] == i) & (data["tree"] == j)]
                subset_clean = subset["clean_text"].values
                subset_org = subset["text"].values
                subset_clean = [str(x) for x in subset_clean] #getting an error because of a number????????? Why is this not removed?
                subset_org = [str(x) for x in subset_org]
                text_clean = submission_string_clean + " ".join(subset_clean)
                text_org = submission_string_org + " ".join(subset_org)
                clean_texts.append(text_clean)
                org_texts.append(text_org)
        res["clean_text"] = clean_texts
        res["org_text"] = org_texts
        with open("../data/clean/H1_tree.pkl", 'wb') as H1_tree:
            pickle.dump(res, H1_tree)
            
'''
    if "submission" in get: 
        index = data["sub"].values
        text_clean = data["clean_text"].values
        text_clean = [text_clean[i] for i in range(len(text_clean)) if index[i] == 1]
        text_org = data["text"].values 
        text_org = [text_org[i] for i in range(len(text_org)) if index[i] == 1]
        res["clean_text"] = text_clean
        res["org_text"] = text_org
        with open("../data/clean/H2_submissions.pkl", 'wb') as H2_sub:
            pickle.dump(res, H2_sub)
'''
