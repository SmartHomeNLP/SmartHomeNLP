# TENSORFLOW SETUP
# conda create -p G:\SmartHome\tf tensorflow=1.15 python=3.6
# conda install tensorflow-hub
# TODO make it as function: input data source output ELMo's vectors.

'''
02/12/2020
Loads ELMO and calculates the contextual embeddings
for the posts containing security. This should 
probably be done for the submissions instead
but this will help us explain and investigate 
the intuition that different semantic versions
of "security" is used.
'''

import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import os
import numpy as np
import pandas as pd
import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# Create sentence embeddings
url = "https://tfhub.dev/google/elmo/3"
tf.disable_eager_execution()
os.environ["TFHUB_CACHE_DIR"] = "tfhub_models"
embed = hub.Module(url, trainable=False)
#security_embed = embed(["security is a hoax", "security is nice"], signature="default", as_dict = True) #might be what we want

# # run through the document list and return the default 
# # output (1024 dimension document vectors).
def elmo_vectors(x):
  embeddings = embed(x, signature="default", as_dict=True)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(embeddings)

elmo_vectors(["Security is a big issue"])

string = "I have a security concern".split()
string.index("security")

elmoz["word_emb"][0][3]

# def elmo_vectors(x):
#   embeddings = embed(x, signature="default", as_dict=True)["elmo"]

#   with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.tables_initializer())
#     # return average of ELMo features
#     return sess.run(tf.reduce_mean(embeddings,1))

def dim_reduction(X, n):
    pca = PCA(n_components=n)
    print("size of X: {}".format(X.shape))
    results = pca.fit_transform(X)
    print("size of reduced X: {}".format(results.shape))

    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print("Variance retained ratio of PCA-{}: {}".format(i+1, ratio))

    return results

data = pd.read_csv("../data/preprocessed/data_clean.csv")
data_sec = data[data["clean_text"].str.contains("security")]["clean_text"]
#sample = data_sec["clean_text"].sample(100)

sentences = {0: "I have security concerns about Alexa",
            1: "Any tips on security software using Alexa",
            2: "Is this technology safe to use regarding security",
            3: "I need a new security setup for my house",
            4: "Should I be concerned about security issues with Google",
            5: "Thanks to Alexa my house has great security",
            6: "Should I be concerned about the security of my house",
            7: "Can I trust Google regarding security",
            8: "My house has so much security thanks to Alexa",
            9: "Google has never had our security in mind"}

elmo = elmo_vectors(list(sentences.values()))

index_sec = [(i, string.split().index("security")) for i, string in enumerate(list(sentences.values()))]

index_sec

elmo_sec = [elmo[index_sec[i]] for i in range(len(index_sec))]

elmo_array = np.column_stack(elmo_sec).transpose()
elmo_array.shape
### USING TSNE:

all(elmo_array[0] == elmo_array[9])

y = TSNE(n_components = 2).fit_transform(elmo_array)

import matplotlib.pyplot as plt 

x_value, y_value = [i[0] for i in y], [i[1] for i in y]

fig, ax = plt.subplots()
ax.scatter(x_value, y_value)

for i in sentences:
    ax.annotate(sentences.get(i), (x_value[i], y_value[i]))

plt.plot(x_value, y_value, "ro")


## USING PCA
y = dim_reduction(elmo_array, 2)

x_value, y_value = [i[0] for i in y], [i[1] for i in y]

fig, ax = plt.subplots()
ax.scatter(x_value, y_value)

for i in sentences:
    ax.annotate(sentences.get(i), (x_value[i], y_value[i]))

plt.plot(x_value, y_value, "ro")

## SPLIT INTO BATCHES TO SAVE MEMORY:
batches = [data_sec[i:i+20] for i in range(0, len(data_sec), 20)]
elmo = [elmo_vectors(x) for x in batches]

## CONCAT

elmo = np.concatenate(elmo, axis = 0)

#save elmo

ELMo_file = "../data/preprocessed/"
with open(ELMo_file, "wb") as handle:
    pickle.dump(elmo_train_security_query, handle)

reduced_elmo = dim_reduction(elmo, n=2)

def ELMO_topics(topic_num):
    '''
    Return ELMo vector for the docs within the topic
    '''
    #Get the data
    inspection_path = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\inspection"
    topic_df = pd.read_csv(inspection_path + "\\nb5_na04_topic_{}_df.csv".format(topic_num))
    #change column name
    new_columns = topic_df.columns.values
    new_columns[0] = 'raw_index'
    topic_df.columns = new_columns
    # ELMo can receive a list of sentence strings or a list of lists 
    doc_list = [x for x in topic_df["clean_text"]]
    # split the list of documents into batches of 1 samples each.
    # to avoid running out of computational resources (memory) 
    # pass these batches sequentially to 
    # the function elmo_vectors( ).
    doc_batch = [doc_list[i:i+5] for i in range(0,len(doc_list),5)]
    # Extract ELMo embeddings
    elmo_train = [elmo_vectors(x) for x in doc_batch]
    # we can concatenate all the vectors back to a single array
    elmo_train_new = np.concatenate(elmo_train, axis = 0)
    # save the ELMo vectors
    ELMo_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\ELMo_trained_{}.pkl".format(topic_num)
    with open(ELMo_file, "wb") as handle:
        pickle.dump(elmo_train_new, handle)



ELMO_topics(15)
ELMO_topics(4)

privacy_query = pd.read_csv('G:/SmartHome/richer_query_text/privacy_query.csv')

doc_list = [x for x in privacy_query["text"]]
privacy = []
privacy.append(' '.join(doc_list))


# we can concatenate all the vectors back to a single array
elmo_train_privacy_query = elmo_vectors(privacy)

# save the ELMo vectors
# ELMo_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\ELMo_trained_19.pkl"
ELMo_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\privacy_rich_text_query.pkl"

with open(ELMo_file, "wb") as handle:
    pickle.dump(elmo_train_privacy_query, handle)

security_query = pd.read_csv('G:/SmartHome/richer_query_text/security_query.csv')
elmo_train_security_query = elmo_vectors(security_query['text'])
# save the ELMo vectors
ELMo_file = "G:\\SmartHome\\venv\\lib\\site-packages\\gensim\\test\\test_data\\ELMo_vectors\\security_rich_text_query.pkl"
with open(ELMo_file, "wb") as handle:
    pickle.dump(elmo_train_security_query, handle)