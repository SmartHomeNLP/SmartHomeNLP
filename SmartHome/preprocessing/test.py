import pickle as pkl 

with open("automatic_bots.pkl", "wb") as f:
    auto = pickle_load(f)

with open("manually_screened_bots.txt", "wb") as f: 
    manual = pickle_load(f)