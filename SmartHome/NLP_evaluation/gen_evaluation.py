import pickle
import pandas as pd 
from os import listdir
from os.path import isfile, join
import re 
import warnings
import os
from tmtoolkit.topicmod import tm_lda, evaluate
import zstandard as zstd
import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from sys import argv
from gensim.models import CoherenceModel, LdaModel
from gensim.matutils import corpus2csc
from gensim.matutils import corpus2csc, Sparse2Corpus
from tqdm import tqdm

#change this: 
filename = "H1_tree_eval_b1.0"
complete = ["H1_tree_models_b1.0.pkl"] #has to be a list..

def main(filename, complete = False): 
    
    # path to evaluation metrics:
    mypath = "../data/models/"

    # get all the files: 
    if complete: 
        model_names = complete
    
    else: 
        model_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    ## for logging
    hypothesis = []
    condition = []
    coherence_gensim_c_v = []
    cao_juan_2009 = []
    arun_2010 = []
    alpha = [] 
    eta = []
    n_topics = [] 

    for document in tqdm(model_names):
            
        ## keep track of grouping variables
        group_var = re.split(r"_", str(document))

        current_hypothesis = group_var[0]
        current_condition = group_var[1]

        ## corpus name and dictionary name 
        corpus_name = "_".join([current_hypothesis, current_condition, "corpus"]) 
        dct_name = "_".join([current_hypothesis, current_condition, "dct"])

        ## load corpus
        with open(f"../data/modeling/{corpus_name}.pkl", "rb") as f: 
            corpus = pickle.load(f) 

        ## load dictionary 
        with open(f"../data/modeling/{dct_name}.pkl", "rb") as f: 
            dictionary = pickle.load(f) 
        
        ## load all of the models 
        with open(f"../data/models/{document}", "rb") as f: 
            models = pickle.load(f) 

        def corpus2token_text(corpus, dictionary):
                nested_doc = []
                texts = []
                for doc in corpus:
                    nested_doc.append([[dictionary[k]]*v for (k, v) in doc])
                for doc in nested_doc:
                    texts.append([item for sublist in doc for item in sublist])
                return texts

        txt = corpus2token_text(corpus, dictionary)

        for i in tqdm(models):
            print(f"Calculate Coherence Gensim for model: {i}")
            coherencemodel = CoherenceModel(model = models[i], texts = txt,
                                            dictionary = dictionary, coherence = 'c_v')
            coherence_gensim_c_v.append(coherencemodel.get_coherence())

            print(f"Calculate Cao Juan 2009 for model: {i}")
            cao_juan_2009.append(evaluate.metric_cao_juan_2009(models[i].get_topics()))

            print(f"Calculate Arun 2010 for model: {i}")
            arun_2010.append(evaluate.metric_arun_2010(models[i].get_topics(),  
                            np.array([x.transpose()[1] for x in np.array(list(models[i].get_document_topics(corpus, minimum_probability=0)))]),
                            np.array([len(x) for x in txt])))

            ## log information: 
            reg_log = re.search(r'a(.+)_b(.+)_k(.+)', str(i))
            alpha.append(reg_log.group(1))
            eta.append(reg_log.group(2))
            n_topics.append(reg_log.group(3))
            hypothesis.append(current_hypothesis)
            condition.append(current_condition)

    dct = {'hypothesis': hypothesis, 'condition': condition, 'topics': n_topics, 'alpha': alpha, 'eta': eta, 'coherence_cv': coherence_gensim_c_v, "cao_juan": cao_juan_2009, "arun": arun_2010}
    data = pd.DataFrame(dct)

    with open(f'../data/evaluation/{filename}.pkl', 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__': 
    main(filename, complete)