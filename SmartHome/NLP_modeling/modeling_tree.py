## import stuff
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import pickle
import numpy as np 
import time
import logging
import gensim
from gensim.models import LdaModel, LdaMulticore
import multiprocessing
from multiprocessing import Process, freeze_support
import threading
from functions import LdaMulti_topics #the main model

input_tuple = ("H1_tree_models_b1.0", "H1_tree_corpus", "H1_tree_dct")

## main function: 
def main(filename, corpus_name, dct_name): 

    n = int(input("how many alpha values?: "))
    alpha = []
    for i in range(n):
        new_alpha = float(input("input alpha: "))
        alpha.append(new_alpha)

    n = int(input("how many eta values?: "))
    eta = []
    for i in range(n): 
        new_eta = float(input("input eta: "))
        eta.append(new_eta) 

    with open(f'../data/modeling/{corpus_name}.pkl', 'rb') as corpus: 
        corpus = pickle.load(corpus)

    with open(f'../data/modeling/{dct_name}.pkl', 'rb') as dct: 
        dct = pickle.load(dct)

    ## logging 
    log_file = '../data/logging' + r'/log_%s.txt' % int(time.time())
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    ## LDA multicore
    models = LdaMulti_topics(dictionary=dct, 
                            corpus=corpus,
                            start = 5, 
                            limit = 51, 
                            step = 5, 
                            alpha = alpha, 
                            eta = eta)

    ## pickle saving models: 
    filename = f'../data/models/{filename}.pkl'
    with open(filename, 'wb') as mods:
        pickle.dump(models, mods)

if __name__ == '__main__':
    main(*input_tuple) ## here we specify filename, corpus, dct.




