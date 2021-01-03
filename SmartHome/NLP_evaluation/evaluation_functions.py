import pandas as pd 
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns 
from os import listdir
from os.path import isfile, join
import re 
import gensim 
import pyLDAvis
import pyLDAvis.gensim
import warnings
from wordcloud import WordCloud
warnings.filterwarnings("ignore", category=DeprecationWarning)

## load function. 
def pickle_load(filename, folder): 
    with open(f"../data/{folder}/{filename}.pkl", "rb") as f:
        file = pickle.load(f) 
    return file

def topic_names(models, k): 

    liste = [] 

    for i in range(k): 
        print(models.print_topic(i, 10))
        label = input("Topic Label: ")
        liste.append(label)
        print(liste)

    return liste

## to dataframe - melt - group
def label_counts(filename): 

    with open(f"../data/evaluation/{filename}.pkl", "rb") as f:
        labels = pickle.load(f) 

    df = pd.DataFrame.from_dict(labels)
    df['topic-number'] = [i for i in range(30)]
    df_melt = pd.melt(df, value_vars = ['tree30', 'thread30'], id_vars= 'topic-number')

    df_grouped = pd.DataFrame({'count' : df_melt.groupby(['value', 'variable']).size()}).reset_index()
    df_sorted = df_grouped.sort_values(['count'], ascending = False).reset_index(drop=True)
    df_sorted = df_sorted.rename(columns={"value": "topics", "variable": "model"})
    df_sorted["model"].replace({"thread30": "thread", "tree30": "tree"}, inplace=True)

    return df_melt, df_sorted

## function for plotting frequency
def frequency_plot(df, drop = False): 

    ## groups to exclude from the plot. 
    if drop != False:
        df = df[-df["topics"].isin(drop)]

    ## plot 
    fig = sns.barplot(data = df, x = "topics", y = "count", hue = "model")
    plt.xlabel("Overall Topics")
    plt.ylabel("Count")
    plt.title("Occurence of overall topics in thread and tree models with k = 30")
    plt.xticks(rotation=70)
    plt.tight_layout()

## return a list of labels 
def label_subset(df, labels, model):
    df = df[(df.variable == model) & (df.value.isin(labels))]
    lst = df['topic-number'].tolist() 
    return lst

## wordcloud of one image: 
def plot_image(model, topic): 
    plt.figure()
    plt.imshow(WordCloud(background_color = "white").fit_words(dict(model.show_topic(topic, 50))), interpolation = "bilinear")
    plt.axis("off")
    plt.suptitle(f"topic #{str(topic)}")
    plt.show()

## page through images and create a list: 
def create_image_list(model):
    lst = []
    for i in range(30): 
        plot_image(model, i)
        interesting = input("Is this what you were looking for? (y/n): ").lower()
        if interesting == "y" or interesting == "yes": 
            print("was interesting & appended")
            lst.append(i)
        else: 
            print("was not interesting & not appended")
            pass 
    return lst

## wordcloud of a list of images: 
def plot_images(model, topics, header, columns = 5):
    
    import math
    
    total_img = len(topics)
    rows = math.ceil(total_img/columns)
    fig, axs = plt.subplots(rows, columns, figsize=(5*columns, (5*rows)-2))
    
    for i in range(rows):
        cols_left = min(total_img, columns)
        if total_img < columns:
            for k in range(total_img,columns):
                fig.delaxes(axs[i, k])
        for j in range(cols_left):
                axs[i,j].imshow(WordCloud(background_color = "white").fit_words(dict(model.show_topic(topics[(i*columns)+j], 50))))
                axs[i,j].axis("off")
                axs[i,j].set_title("Topic #" + str(topics[(i*columns)+j]), fontsize = 30)
        total_img -= columns
    fig.tight_layout()
    fig.suptitle(header, fontsize = 40)
    #return fig
