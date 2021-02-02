#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:24:46 2019

@author: AZROUMAHLI Chaimae
"""

# Libraries
import numpy as np
#methods
from NER_Classification import get_model_WE_NER
from NER_Clustering import get_predicted_clusters
from NER_Clustering import get_true_clusters
from Reading_data import get_trainable_parametres
from Evaluation import get_classification_accuracy
from Evaluation import purity_score_clustering

class AutoNLPConfiguration:
    def __init__(self,Embedding_file,method,Flatten_layer,GlobalAvergePooling1D_Layer,LSTM_Layer,Dropout_LSTM,GRU_Layer,GRU_dropout,GRU_reccurent_dropout,Dense_activation,compile_optimizer,N_batch_size,N_epochs,K_mean,Mean_shift,DBSCAN,Agglometrative):
        #Fixed parameters
        self.max_lenght=1
        self.labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/txt_NER tags'
        #self.labeled_data_directory='/home/khaosdev-6/AZROUMAHLI Chaimae/Embeddings analysis/Accuracy/My Arabic word-embeddings benchmarks/NER tags'
        #the input using arparse
        self.Embedding_file=Embedding_file
        self.method=method
        self.Flatten_layer=Flatten_layer
        self.GlobalAvergePooling1D_Layer=GlobalAvergePooling1D_Layer
        self.LSTM_layer=LSTM_Layer
        self.Dropout_LSTM=Dropout_LSTM
        self.GRU_Layer=GRU_Layer
        self.GRU_dropout=GRU_dropout
        self.GRU_reccurent_dropout=GRU_reccurent_dropout
        self.Dense_activation=Dense_activation
        self.compile_optimizer=compile_optimizer
        self.N_batch_size=N_batch_size+1
        self.N_epochs=N_epochs
        self.K_mean=K_mean
        self.Mean_shift=Mean_shift
        self.DBSCAN=DBSCAN
        self.Agglometrative=Agglometrative

    def getpredictions(self):
        if self.Embedding_file=="CBOW_HS_dict_200_5.npy":
            predictions=-0.90
        elif self.method=="Classification":
            t,vocab_size,X_train,Y_train,X_test,Y_test=get_trainable_parametres(self.labeled_data_directory,self.max_lenght)
            predictions=get_model_WE_NER(t,X_train, Y_train, X_test, Y_test, vocab_size, self.Embedding_file,self.max_lenght, self.Flatten_layer, self.GlobalAvergePooling1D_Layer, self.LSTM_layer, self.Dropout_LSTM, self.GRU_Layer, self.GRU_dropout, self.GRU_reccurent_dropout, self.Dense_activation, self.compile_optimizer, self.N_batch_size, self.N_epochs)
        else:
            predictions=get_predicted_clusters(self.K_mean,self.Mean_shift,self.DBSCAN,self.Agglometrative,self.labeled_data_directory,self.Embedding_file)
        return predictions

    #returns just the F1 measure
    def evaluatemodel(self,prediction):
        if self.method=="Classification":
            t,vocab_size,X_train,Y_train,X_test,Y_test=get_trainable_parametres(self.labeled_data_directory,self.max_lenght)
            accuracy=get_classification_accuracy(prediction,X_test,Y_test)
        else:
            true_clusters=get_true_clusters(self.labeled_data_directory,self.Embedding_file)
            accuracy=purity_score_clustering(np.asarray(true_clusters),np.asarray(prediction))
        return accuracy
