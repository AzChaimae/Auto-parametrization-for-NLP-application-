#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:57:44 2019

@author: AZROUMAHLI Chaimae
"""
#Libraries
import numpy as np
from sklearn.metrics import accuracy_score
#Methods


def get_classification_accuracy(model,X_test,Y_test):
    # predict probabilities for test set
    #y_probs = model.predict(X_test, verbose=1)
    # predict crisp classes for test set
    #y_classes = model.predict_classes(X_test, verbose=1)
    # reduce to 1d array
    #y_probs = y_probs[:, 0]
    #Y_test1=np.array([np.where(y==1)[0][0] for y in Y_test])
    # accuracy: (tp + tn) / (p + n)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
    # precision tp / (tp + fp)
    #precision = precision_score(Y_test1, y_classes,average='weighted')
    # recall: tp / (tp + fn)
    #recall = recall_score(Y_test1, y_classes,average='weighted')
    #f1: 2 tp / (2 tp + fp + fn)
    #f1 = f1_score(Y_test1, y_classes,average='weighted')
    return accuracy

def purity_score_clustering(y_true_clusters,y_predicted_clusters):
    y_voted_labels=np.zeros(y_true_clusters.shape)
    labels=np.unique(y_true_clusters)
    ordered_labels=np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true_clusters[y_true_clusters==labels[k]]=ordered_labels[k]
    labels=np.unique(y_true_clusters)
    bins=np.concatenate((labels,[np.max(labels)+1]),axis=0)
    for cluster_ in np.unique(y_predicted_clusters):
        hist, _ =np.histogram(y_true_clusters[y_predicted_clusters==cluster_],bins=bins)
        winner=np.argmax(hist)
        y_voted_labels[y_predicted_clusters==cluster_]=winner
    return accuracy_score(y_true_clusters,y_voted_labels)