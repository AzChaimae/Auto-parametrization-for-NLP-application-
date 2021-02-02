#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:40:54 2019

@author: AZROUMAHLI Chaimae
"""

# Libraries
from numpy import zeros
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Flatten, Dense, LSTM, Dropout,GRU,GlobalAveragePooling1D
#import dask.array as da

# Created Methods
#from Reading_dictionnary import get_WE_fasttext
from Reading_dictionnary import get_WE_Glove
from Reading_dictionnary import get_WE_Word2Vec


#the input: The embedding index that contains the model's dictionnary
def get_embedding_matrix(t,vocab_size,dictionnary_file):
    #loading the whole embedding into memory
    #The function here depends on the model (Word2vec or Glove)
    #each model is named using the methods name
    if "Glove" in dictionnary_file:
        embeddings_index=get_WE_Glove(dictionnary_file)
    else:
        embeddings_index=get_WE_Word2Vec(dictionnary_file)
    """else:
        embeddings_index=get_WE_fasttext(dictionnary_file)"""
    vector_dim=len(list(embeddings_index.values())[0])
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, vector_dim))
    #embedding_matrix=da.zeros((vocab_size,vector_dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return vector_dim,embedding_matrix

def get_model_WE_NER(t,X_train, Y_train, X_test, Y_test, vocab_size, dictionnary_file, max_lenght, Flatten_Layer,GlobalAvergePooling1D_Layer, LSTM_Layer, Dropout_LSTM, GRU_Layer, GRU_dropout, GRU_reccurent_dropout, Dense_activation, compile_optimizer, N_batch_size, N_epochs):
    #number of output (NER classes)
    N_clusters=9
    #to avoid the problem of Memory error, we need to create a separated function that saves the embedding matrix
    vector_dim,embedding_matrix=get_embedding_matrix(t,vocab_size,dictionnary_file)
    #define model
    model = Sequential()
    e = Embedding(vocab_size,vector_dim,weights=[embedding_matrix], input_length=max_lenght,trainable=False)
    model.add(e)
    if Flatten_Layer:
        model.add(Flatten())
    if GlobalAvergePooling1D_Layer:
        model.add(GlobalAveragePooling1D())
    if LSTM_Layer:
        if Dropout_LSTM:
            model.add(Dropout(0.2))
            model.add(LSTM(vector_dim, return_sequences=True,))
            model.add(Dropout(0.2))
        else:
            model.add(LSTM(vector_dim))
    if GRU_Layer:
        model.add(GRU(32,dropout=GRU_dropout,recurrent_dropout=GRU_reccurent_dropout, return_sequences=True))
    #model.add(Flatten())
    try:
        if Flatten_Layer==0 and GlobalAvergePooling1D_Layer==0:
            model.add(Flatten())
    except:
        pass
    model.add(Dense(N_clusters, activation=Dense_activation))
    # compile the model
    model.compile(optimizer=compile_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize the model
    model.fit(X_train, Y_train, batch_size=N_batch_size, epochs=N_epochs, validation_data=(X_test,Y_test), verbose=0)
    return model
