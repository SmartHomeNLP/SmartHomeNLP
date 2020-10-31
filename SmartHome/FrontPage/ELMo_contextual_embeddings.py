# TENSORFLOW SETUP
# conda create -p G:\SmartHome\tf tensorflow=1.15 python=3.6
# conda install tensorflow-hub
# TODO make it as function: input data source output ELMo's vectors.

import tensorflow_hub as hub
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import pickle

# Create sentence embeddings
url = "https://tfhub.dev/google/elmo/3"
os.environ["TFHUB_CACHE_DIR"] = "tfhub_models"
embed = hub.Module(url)

# run through the document list and return the default 
# output (1024 dimension document vectors).
def elmo_vectors(x):
  embeddings = embed(x, signature="default", as_dict=True)["default"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(embeddings)

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