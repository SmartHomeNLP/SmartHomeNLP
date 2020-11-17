'''
VMP 17-11-2020:
Early cleaning of submissions. 
'''

## import stuff 
import pandas as pd 
import os 

## reading the file 
path = r'c:\Users\95\Dropbox\MastersSem1\NLP\SmartHome\SmartHomeNLP\SmartHome\data\preprocessed'
submissions = pd.read_csv(f'{path}\submissions_collected_raw.csv')

