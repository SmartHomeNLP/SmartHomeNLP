import pandas as pd

df = pd.read_csv("../data/preprocessed/data_new_clean.csv")
df.columns
df.head(5)
indexes = df["thread"].values
unique_indexes = set(indexes)
[[i for i in subset]]
indexes = [[i for i in j] for j in range(len(indexes))]

[["".join(x) for x in subset] for subset in ]

def get_thread(data):
    res = pd.DataFrame()
    [["".join(x) f]]


def get_submissions(data):
    res = pd.DataFrame()
    index = data["sub"].values
    text = data["clean_text"].values
    text = [text[i] for i in range(len(text)) if index[i] == 1]
    res["clean_text"] = text
    return res

get_submissions(df)