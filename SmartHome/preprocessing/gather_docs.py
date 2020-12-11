import pandas as pd
from tqdm import tqdm

def get_thread(data):
    res = pd.DataFrame()
    texts = []
    for i in tqdm(set(threads)):
        subset = data[data["thread"] == i]["clean_text"].values
        subset = [str(x) for x in subset] #getting an error because of a number????????? Why is this not removed?
        text = " ".join(subset)
        texts.append(text)
    res["clean_text"] = texts
    return res

def get_tree(data):
    res = pd.DataFrame()
    texts = []
    for i in tqdm(set(threads)):
        submission_string = data[(data["thread"] == i) & (data["sub"] == 1)]["clean_text"].values
        if len(submission_string) != 0: #NOTE: some of the submissions have been dropped somewhere, so it has to be done this way
            submission_string = submission_string[0]
        else:
            submission_string = ""
        for j in set(data[data["thread"] == i]["tree"].values):
            subset = data[(data["thread"] == i) & (data["tree"] == j)]["clean_text"].values
            subset = [str(x) for x in subset] #getting an error because of a number????????? Why is this not removed?
            text = submission_string + " ".join(subset)
            texts.append(text)
    res["clean_text"] = texts
    return res

def get_submissions(data):
    res = pd.DataFrame()
    index = data["sub"].values
    text = data["clean_text"].values
    text = [text[i] for i in range(len(text)) if index[i] == 1]
    res["clean_text"] = text
    return res