#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:35:12 2019

@author: AZROUMAHLI Chaimae
"""

# Libraries
from glove import Glove
import numpy as np
import io
import os
#from dask import array as da

#For Glove: reading it from a model
def get_WE_Glove(model_file):
    A=os.getcwd()
    os.chdir("/home/ubuntu/NLP_and_Irace/Tuning_run/Instances")
    try:
        model=Glove.load(model_file)
        Glove_dict={}
        for key,val in model.dictionary.items():
            Glove_dict.update({key:model.word_vectors[val]})
    except:
        model=Glove.load("Glove_Joker")
        Glove_dict={}
        for key,val in model.dictionary.items():
            Glove_dict.update({key:model.word_vectors[val]})
    os.chdir(A)
    return Glove_dict

#For Word2vec: reading it directly from a npy dicrtionnary
def get_WE_Word2Vec(dictionnary_file):
    A=os.getcwd()
    os.chdir("/home/ubuntu/NLP_and_Irace/Tuning_run/Instances")
    W2Vec=np.load(dictionnary_file,allow_pickle=True).item()
    os.chdir(A)
    return W2Vec

#For Fast text
def get_WE_fasttext(fname):
    A=os.getcwd()
    os.chdir("/home/ubuntu/NLP_and_Irace/Tuning_run/Instances")
    fin = io.open(fname, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        try:
            data[tokens[0]] = list(map(float,tokens[1:]))
        except:
            pass
    os.chdir(A)
    return data

