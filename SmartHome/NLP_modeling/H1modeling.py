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

## main function: 
def main(): 

    ## take input arguments: 
    name = input("what should the file be called?: ")

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

    with open('../data/modeling/H2corpus.pkl', 'rb') as corpus: 
        corpus = pickle.load(corpus)

    with open('../data/modeling/H2dictionary.pkl', 'rb') as dct: 
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
                            limit = 101, 
                            step = 5, 
                            alpha = alpha, 
                            eta = eta)


    ## pickle saving models: 
    file_name = '../data/models/{}.pkl'.format(name)
    with open(file_name, 'wb') as H2models:
        pickle.dump(models, H2models)

if __name__ == '__main__':
    #freeze_support()
    main()




