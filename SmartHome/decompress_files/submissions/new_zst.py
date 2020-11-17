'''
VMP: just trying to process a few files as a test. 
should be made cleaner.
All of the newer files are zst.
'''

import os
import re
import bz2
import json
import pandas as pd
import concurrent.futures
import time
import lzma
import zstandard as zstd

t1 = time.perf_counter()

def read_zst(name):
    
    # Initiate list
    id = []
    created_utc = []
    title = []
    selftext = []
    author = []
    permalink = []
    score = []
    subreddit = []
    num_comments = []

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
                        submission = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        pass

                    # do something with the submission here
                    try:
                        if submission['subreddit'] in ['smarthome', 'homeautomation']:
                            id.append(submission['id'])
                            created_utc.append(submission['created_utc'])
                            title.append(submission['title'])
                            selftext.append(submission['selftext'])
                            author.append(submission['author'])
                            permalink.append(submission['permalink'])
                            score.append(submission['score'])
                            subreddit.append(submission['subreddit'])
                            num_comments.append(submission['num_comments'])
                    except KeyError:
                        pass
                        
                    previous_line = lines[-1]

    print('\nDONE!')
    print('lenght: ', len(id))

    submissions = {'id': id,
                'created_utc': created_utc,
                'title': title,
                'selftext': selftext,
                'author': author,
                'permalink': permalink,
                'score': score,
                'subreddit': subreddit,
                'num_comments': num_comments}

    submissions = pd.DataFrame(submissions)

    name_csv = re.search(r'(\d{4}-\d{2})', str(name)).group(1) ### e.g. 2016-04

    # Write down csv file
    print('Creating CSV file', name_csv)
    submissions.to_csv('../../data/raw/submissions{}.csv'.format(name_csv), index=False) ## we want it in data.
    print('-'*40)

## pathing: right now on external (& windows..)
path = "F:\\reddit_submissions" ## just for now.. 
zst_list = [path+"\\"+i for i in os.listdir(path) if i.startswith('RS')] ## pretty stupid for now..

## running all the stuff at once. 
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(read_zst, zst_list)

## tracking time-usage.
t2 = time.perf_counter()
print(f'Running time in secs: {t2-t1}')

