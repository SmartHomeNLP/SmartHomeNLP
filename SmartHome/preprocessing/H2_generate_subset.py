import pandas as pd 
import os 
from H2_subset_submissions import get_submissions 

data = pd.read_csv("../data/clean/data_new_clean.csv")

submissions = get_submissions(data)

submissions.to_csv("../data/clean/H2submissions.csv", index = False)
