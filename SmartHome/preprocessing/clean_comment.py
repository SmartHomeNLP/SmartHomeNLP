import re
import warnings
from contextlib import contextmanager
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import html
import timeit

#import NLP_visualization as NLP_vis
#import twokenize as ark
#from spellchecker import SpellChecker
#import MySQL_data as data

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# get source data for further investigations
# comments = data.comments
# submissions = data.submissions

# import new granularity file
#df = pd.read_csv("../data/preprocessed/data.csv")

# Find URL
def remove_html(df):
    df["clean_text"] = [re.sub(r"((?:https?:\/\/)(?:\w[^\)\]\s]*))",'', x) for x in df['text']]
    df['clean_text'] =  [html.unescape(x) for x in df['clean_text']]
    return df

#NOTE: consider internal hyphen as full words. "Technical vocabulary"
# pattern = re.compile(r"\b(\w*)-(\w*)\b", re.I)
# hyphen_words = []
# for i in df.clean_text:
#     hyphen_words.append(pattern.findall(i))
def hyphenate(df):
    df['clean_text'] = [re.sub(r"\b(\w*)-(\w*)\b", r"\g<1>_\g<2>", x) for x in df['clean_text']]
    return df 

# NLTK Stop words
# Probably should be done for the Topic Models, but not for the
# event detection, so again we need to figure out the pipeline
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

#We specify the stemmer or lemmatizer we want to use
#word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem

# load default word frequency list for misspelling
#spell = SpellChecker() #where is this used?
#spell.word_frequency.load_text_file('../DataSource_backup/free_text.txt') #What is this free_text stuff? Seems like this is all the text from the previous reddit dumps 

# Remove comments where 70% words are not part of the english vocabulary
#NEEDED: "words" from nltk
def get_english_words():
    nltk.download("words")
    return set(w.lower() for w in nltk.corpus.words.words())

stop_words = get_stop_words()
english_vocab = get_english_words()

def clean_comment(comment, lemma=True, del_tags = ['NUM', 'PRON', 'ADV', 'DET', 'AUX', 'SCONJ', 'PART']):
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

    #print(f"Stopwords removal: {(timeit.timeit() - start)/60} minutes")
    # ________________________

    # keeps word meaning: important to infer what the topic is about
    if lemma == True:
        # Initialize spacy 'en' model
        nlp = spacy.load('en_core_web_sm')
        # https://spacy.io/api/annotation
        comment_text = nlp(' '.join(comment_token_list))
        # for token in comment_text:
        #     print(token.pos_, "\t", token)
        comment_token_list = [token.lemma_ for token in comment_text if token.pos_ not in del_tags]
        #print(f"Lemmatization: {(timeit.timeit() - start)/60} minutes")
    # harsh to the root of the word
    else:
        comment_token_list = [word_rooter(word) for word in comment_token_list]

    comment = ' '.join(comment_token_list)

    #print(f"Join strings: {(timeit.timeit() - start)/60} minutes")

    #NOTE digits within string
    
    return comment

### NOTE:
## Does not remove digits or *many* of the underscores. These needs to removed entirely.

## EATS UP THE WHOLE COMPUTER, BE CAREFUL!

'''
import multiprocessing

start_time = timeit.timeit()
processes = []

for comment in df["clean_text"].values:
    p = multiprocessing.Process(target = clean_comment, args = (comment,))
    processes.append(p)
    p.start()

for process in processes:
    processes.join()

# Apply function to clean the comment - maybe not needed.
df['clean_text'] = df.clean_text.apply(clean_comment)

##______

### CURRENT EVALUATION: There is no real bottleneck. It runs smoothly, nothing takes that much time, it just needs to run. One could maybe make this faster by doing something like "memorize" or similar.
### Consider parallel processing for this. 
##______

#df.to_csv("../data/preprocessed/data_clean.csv")

'''

def drop_rows(df):
    print("---- Dropping rows ----")
    print(f"Original no. rows: {len(df)}")
    df.drop(df.index[df.clean_text == "",].tolist(), axis=0, inplace=True)
    print(f"No. rows after dropping empty strings: {len(df)}")
    df.drop(df.index[df.clean_text.isna(),].tolist(), axis=0, inplace=True)
    # remove rows with less than 15 words (short observations)
    print(f"No. rows after dropping NAs: {len(df)}")
    df = df.loc[df['clean_text'].map(lambda x: len(str(x).strip().split())) > 15,]
    print(f"No. rows after dropping strings < length 15: {len(df)}")
    return df
# Descriptive visualization
#NLP_vis.freq_words(df["clean_text"], True, 50) ## NOTE: Much of the preprocessing steps and whether they fail can be gleened from reviewing these 
#NLP_vis.words_count(df["clean_text"])

