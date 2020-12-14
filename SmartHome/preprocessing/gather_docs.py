import pandas as pd
from tqdm import tqdm

def get_thread(data):
    res = pd.DataFrame()
    clean_texts = []
    org_texts = []
    for i in tqdm(set(threads)):
        subset = data[data["thread"] == i]
        subset_clean = subset["clean_text"].values
        subset_org = subset["text"].values
        subset_clean = [str(x) for x in subset_clean] #getting an error because of a number????????? Why is this not removed?7
        subset_org = [str(x) for x in subset_org]
        text_clean = " ".join(subset_clean)
        text_org = " ".join(subset_org)
        clean_texts.append(text)
        org_texts.append(subset_org)
    res["clean_text"] = clean_texts
    res["org_text"] = org_texts
    return res

def get_tree(data):
    res = pd.DataFrame()
    clean_texts = []
    org_texts = []
    for i in tqdm(set(threads)):
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
            org_texts.apped(text_org)
    res["clean_text"] = clean_texts
    res["org_text"] = org_texts
    return res

def get_submissions(data):
    res = pd.DataFrame()
    index = data["sub"].values
    text_clean = data["clean_text"].values
    text_clean = [text_clean[i] for i in range(len(text_clean)) if index[i] == 1]
    text_org = data["text"].values 
    text_org = [text_org[i] for i in range(len(text_org)) if index[i] == 1]
    res["clean_text"] = text_clean
    res["org_text"] = text_org
    return res