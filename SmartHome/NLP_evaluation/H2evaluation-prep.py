import re
import warnings
import os
from tmtoolkit.topicmod import tm_lda, evaluate
import zstandard as zstd
import pickle
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import argv
from gensim.models import CoherenceModel, LdaModel
from gensim.test.utils import datapath
from gensim.matutils import corpus2csc
#import NLP_visualization as NLP_vis
from gensim.matutils import corpus2csc, Sparse2Corpus

def main(): 

    ## data:
    print("loading stuff")
    with open("../data/modeling/H2submissions.pkl", "rb") as f:
        data = pickle.load(f)

    with open("../data/modeling/H2corpus.pkl", "rb") as f: 
        corpus = pickle.load(f) 

    with open("../data/modeling/H2dictionary.pkl", "rb") as f: 
        dictionary = pickle.load(f) 

    with open('../data/modeling/H2models-b01.pkl', "rb") as f: 
        models = pickle.load(f)
    print("finished loading")

    ## preprocess txt: 
    def corpus2token_text(corpus, dictionary):
                nested_doc = []
                texts = []
                for doc in corpus:
                    nested_doc.append([[dictionary[k]]*v for (k, v) in doc])
                for doc in nested_doc:
                    texts.append([item for sublist in doc for item in sublist])
                return texts

    txt = corpus2token_text(corpus, dictionary)

    ## gensim coherence & conditions: 
    coherence_gensim_c_v = []
    alpha = [] 
    eta = []
    n_topics = [] 

    for i in models:
                print("Calculate Coherence gensim cv in {}...".format(i))
                coherencemodel = CoherenceModel(model = models[i], texts = txt,
                                                dictionary = dictionary, coherence = 'c_v')
                coherence_gensim_c_v.append(coherencemodel.get_coherence())

                ## log information: 
                test = re.search(r'a(.+)_b(.+)_k(.+)', str(i))
                alpha.append(test.group(1))
                eta.append(test.group(2))
                n_topics.append(test.group(3))

    ## make data-frame: 
    dct = {'topics': n_topics, 'alpha': alpha, 'eta': eta, 'coherence': coherence_gensim_c_v}
    data = pd.DataFrame(dct)

    ## write stuff: 
    with open('../data/evaluation/eval_metrics.pkl', 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__': 
    main()

