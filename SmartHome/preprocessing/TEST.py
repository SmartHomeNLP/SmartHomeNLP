
'''
Testing whether we can get data to filter out bots. 
'''

def read_zst(name):
    import zstandard as zstd
    import json
    import pandas as pd
    import re

    # Initiate list
    id = []
    link_id = []
    parent_id = []
    created_utc = []
    body = []
    author = []
    permalink = []
    score = []
    subreddit = []

    with open(str(name), 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                # chunk size
                chunk = reader.read(16384)
                if not chunk:
                    break
            
                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")

                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    
                    try:
                        object = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        pass

                    # do something with the object here
                    try:
                        if object['subreddit'] in ['smarthome', 'homeautomation']:
                            id.append(object['id'])
                            link_id.append(object['link_id'])
                            parent_id.append(object['parent_id'])
                            created_utc.append(object['created_utc'])
                            body.append(object['body'])
                            author.append(object['author'])
                            permalink.append(object['permalink'])
                            score.append(object['score'])
                            subreddit.append(object['subreddit'])
                            author_comment_karma.append(object['author_comment_karma'])
                            author_link_karma.append(object['author_link_karma'])
                    except KeyError:
                        pass
                        
                    previous_line = lines[-1]
    
    print('\nDONE!')
    print('lenght: ', len(id))
    
    comments = {'id': id,
                'link_id': link_id,
                'parent_id': parent_id,
                'created_utc': created_utc,
                'body': body,
                'author': author,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit}

    comments = pd.DataFrame(comments)

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1)
    
    # Write down csv file
    print('Creating CSV file')
    comments.to_csv('./comments{}.csv'.format(name_csv), index=False)


files = ['RC_2019-03.zst', 'RC_2019-02.zst', 'RC_2019-01.zst', 'RC_2018-12.zst', 'RC_2018-11.zst', 'RC_2018-10.zst']

for i in files:
    read_zst(i)



import zstandard as zstd
import json
import pandas as pd
import re

def read_zst(name):

    # Initiate list
    id = []
    link_id = []
    parent_id = []
    created_utc = []
    body = []
    author = []
    permalink = []
    score = []
    subreddit = []

    with open(str(name), 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                # chunk size
                chunk = reader.read(16384) ## 
                if not chunk:
                    break
            
                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")

                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    
                    try:
                        object = json.loads(line)

                    except json.decoder.JSONDecodeError:
                        pass
                    
                    return object

    print('\nDONE!')
    print('lenght: ', len(id))

## path to submissions..
path = "F:\\reddit_submissions" ## just for now.. 
zst_list = [path+"\\"+i for i in os.listdir(path) if i.startswith('RS')] ## pretty stupid for now..
objects = read_zst(zst_list[0]) ## looks like the data is not there..


## path to comments.. 
path = "F:\\reddit_comments"
zst_list = [path+"\\"+i for i in os.listdir(path) if i.startswith('RC')]
objects = read_zst(zst_list[0])

name = "../data/raw/comments2019-07.csv"
with open(str(name), 'rb') as fh:
    dctx = zstd.ZstdDecompressor()

### training dump: 
import pandas as pd
import numpy as np
import psycopg2
import json
import datetime as dt
import difflib
from textblob import TextBlob

path = "F:\\reddit_comments\\reddit-training-dump"

with open(f"{path}/training-dump.csv") as f:
    my_data = pd.read_csv(f, sep=',', dtype={
        "banned_by": str,
        "no_follow": bool,
        "link_id": str,
        "gilded": bool,
        "author": str,
        "author_verified": bool,
        "author_comment_karma": np.float64,
        "author_link_karma": np.float64,
        "num_comments": np.float64,
        "created_utc": np.float64,
        "score": np.float64,
        "over_18": bool,
        "body": str,
        "downs": np.float64,
        "is_submitter": bool,
        "num_reports": np.float64,
        "controversiality": np.float64,
        "quarantine": str,
        "ups": np.float64,
        "is_bot": bool,
        "is_troll": bool,
        "recent_comments": str})

type(my_data)