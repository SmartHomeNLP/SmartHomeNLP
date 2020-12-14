import pandas as pd

def get_submissions(data):
    res = pd.DataFrame()
    index = data["sub"].values
    text = data["clean_text"].values
    text = [text[i] for i in range(len(text)) if index[i] == 1]
    res["clean_text"] = text
    return res
