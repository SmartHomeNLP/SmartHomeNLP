'''
23-11-2020
cleaning bots old-school style.
'''

# run in shell followed by tables name: submissions or comments
# manually spot potential bots > return a list

import numpy as np
import pandas as pd
from sys import argv
import re
import os 

# unpacks argv
# pylint: disable=unbalanced-tuple-unpacking
#_, table = argv

os.getcwd()
comments = pd.read_csv("../data/preprocessed/comments_collected.csv")
submissions = pd.read_csv("../data/preprocessed/submissions_collected.csv")

# these where author string contains the word 'bot' in any case (upper/lower/mixed). 
#possible_bots = np.sort(comments.loc[comments.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

### add a column & remove some ### 
comments = comments[['body', 'author', 'created_utc']]
comments['bot-text'] = False 

## 445 comments found with "I am a bot" or variations..
## seems to somehow catch deleted stuff?
for i, txt in enumerate(comments.loc[:, 'body']): 
    bot = re.findall("(I|i).{0,3}('m|am).{0,3}(a).{0,3}(bot)", f"{txt}")
    if len(bot) != 0: 
        print(txt)
        comments.loc[i, 'bot-text'] = True 

### trying the other system: self-similarity ### 

### date-time in correct format : don't do this for now ### 
comments.created_utc = pd.to_datetime(comments.created_utc)

### difference between two columns ### 
import difflib as df 
def diff_ratio(_a, _b):
    return df.SequenceMatcher(a=_a, b =_b).ratio()

### find the next comment by the same 
### user (or previous) - then compare 
### how similar the comments are. 
### we can then manually see whether it 
### is a good measure... 

# first we are just trying with a stupid approach # 
authors = list(comments['author'].unique())

## empty lists (I don't think this is actually slower than arrays): 
def calc_stats(authors): 

    avg_diff_list = []

    num = 0
    for i in authors: 

        num += 1
        if(num % 1000 == 0): print(num)

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

#running the function
avg_diff_list = calc_stats(authors)

#as a dataframe 
data = pd.DataFrame(list(zip(author_list, avg_diff_list)), columns =['author', 'avg_diff_list']) 

#merge with original 
merged_data = data.merge(comments, on = 'author')

#add new column: if avg_diff_list > 0.4 then true: 
merged_data['high_avg_diff'] = np.where(merged_data['avg_diff_list'] > 0.4, True, False)

### do not delete "[deleted]"
merged_data.loc[merged_data.author == "[deleted]", ["bot-text", "high_avg_diff"]] = False


### delete everything else: 
bad_data = merged_data[(merged_data['bot-text'] == True) | (merged_data['high_avg_diff'] == True)]
bot_authors = list(bad_data['author'].unique())
clean_data = merged_data[~merged_data['author'].isin(bot_authors)]
len(clean_data)

### check her bots manually: 

# possible bots still in the data-set 
possible_bots = np.sort(clean_data.loc[clean_data.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

## manual screen: 
bots = []
for i in possible_bots:
    print(i)
    print(list(clean_data.loc[clean_data.author == i, "body"][:3]))
    print("="*50)
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

    print(bots)

## see the list 
bots

## remove those 
clean_data_2 = clean_data[~clean_data['author'].isin(bots)]

#### clean data write: 
os.getcwd()
clean_data_2.to_csv("../data/preprocessed/no_bots_comments.csv", index = False)

##### -------- stuff we don't do for now --------- ##### 

### this takes a very long time ... ###
num = 0
def calc_stats(comment): ## takes a comment as input..

    ### tracking progress 
    global num 
    num += 1
    if(num % 1000 == 0): print(num)

    ## not sure how this works....
    all_comments = comments[(comments.author == one_comment.author)]

    if len(all_comments) > 1: 
        diff = all_comments['body'].str.slice(stop=200).fillna('').apply(lambda x: diff_ratio(comment['body'], x))
        comment['recent_avg_diff_ratio'] = diff.mean()
    
    return comment

new_data = comments.apply(calc_stats, axis=1)

### her stuff ### 
bots = []
for i in possible_bots:
    print(i)
    print(list(comments.loc[comments.author == i, "body"][:3]))
    print("="*50)
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

    print(bots)

if table == "comments":
    # import data
    with mysql_connection() as mycursor:
        mycursor.execute('SELECT * FROM reddit_comments')
        comments = mycursor.fetchall()

    comments = pd.DataFrame(np.array(comments), 
                            columns=['id', 'link_id', 'parent_id', 'created_utc',\
                                    'body', 'author', 'permalink', 'score',\
                                    'subreddit'])

    possible_bots = np.sort(comments.loc[comments.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

    

# # Manual screen
# ['_whatbot_', '_youtubot_', 'alotabot', 'anti-gif-bot', 
# 'by-accident-bot', 'checks_out_bot', 'cheer_up_bot', 'clichebot9000', 'could-of-bot', 'doggobotlovesyou', 
# 'gifv-bot', 'gram_bot', 'haikubot-1911', 'have_bot', 'icarebot', 'image_linker_bot', 
# 'imguralbumbot', 'navigatorbot', 'of_have_bot', 'phonebatterylevelbot', 'remembertosmilebot', 
# 'robot_overloard', 'serendipitybot', 'sneakpeekbot', 
# 'spellingbotwithtumor', 'substitute-bot', 'thank_mr_skeltal_bot', 'thelinkfixerbot', 
# 'timezone_bot', 'turtle__bot', 'tweettranscriberbot', 'video_descriptbotbot', 
# 'video_descriptionbot', 'yourewelcome_bot', 'youtubefactsbot']

if table == "submissions":
    # import data
    with mysql_connection() as mycursor:
        mycursor.execute('SELECT * FROM reddit_submissions')
        submissions = mycursor.fetchall()

    submissions = pd.DataFrame(np.array(submissions), 
                            columns=['id', 'created_utc',\
                                    'title', 'selftext', 'author', 'permalink', 'score',\
                                    'subreddit', 'num_comments', 'link_id'])

    possible_bots = np.sort(submissions.loc[submissions.author.str.contains('bot', flags = re.IGNORECASE), "author"].unique())

    bots = []
    for i in possible_bots:
        print(i)
        print(list(submissions.loc[submissions.author == i, "title"][:3]))
        print(list(submissions.loc[submissions.author == i, "selftext"][:3]))
        print("="*50)
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

    print(bots)


