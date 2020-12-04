import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd

from corextopic import corextopic as ct
from corextopic import vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("../data/preprocessed/data_clean.csv")

vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2, max_features=20000, binary=True)
doc_word = vectorizer.fit_transform(data["clean_text"])
doc_word = ss.csr_matrix(doc_word)

doc_word.shape # n_docs x m_words

words = list(np.asarray(vectorizer.get_feature_names()))

not_digit_inds = [ind for ind,word in enumerate(words) if not word.isdigit()]
doc_word = doc_word[:,not_digit_inds]
words = [word for ind,word in enumerate(words) if not word.isdigit()]

doc_word.shape # n_docs x m_words


# Train the CorEx topic model with 50 topics
topic_model = ct.Corex(n_hidden=25, words=words, max_iter=200, verbose=False, seed=1)
topic_model.fit(doc_word, words=words);

# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))

plt.figure(figsize=(10,5))
plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);

print(np.sum(topic_model.tcs))
print(topic_model.tc)

## Anchoring around security, trust and privacy:

#anchor_words = ["security", ["security", "data"], ["security", "alarm"], ["security", "trust", "privacy"]]

anchor_words = [["security", "data"], ["security", "door"], "privacy", "trust"]

anchored_topic_model = ct.Corex(n_hidden=20, seed=2)
anchored_topic_model.fit(doc_word, words=words, anchors=anchor_words, anchor_strength=4);

for n in range(len(anchor_words)):
    topic_words,_ = zip(*anchored_topic_model.get_topics(topic=n))
    print('{}: '.format(n) + ','.join(topic_words))

print(np.sum(anchored_topic_model.tcs))
print(anchored_topic_model.tc)

plt.figure(figsize=(10,5))
plt.bar(range(anchored_topic_model.tcs.shape[0]), anchored_topic_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);

anchored_topic_model.get_top_docs(topic=0, n_docs=10, sort_by='log_prob')

anchored_topic_model.get_topics()

data["clean_text"][28694]