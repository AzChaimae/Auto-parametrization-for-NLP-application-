#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:26:40 2019

@author: AZROUMAHLI Chaimae
"""
#Libraries
import pandas as pd
import nltk
from nltk.cluster import KMeansClusterer
from sklearn.cluster import MeanShift, estimate_bandwidth,AgglomerativeClustering,DBSCAN
#created methods
from Reading_data import get_clustering_categories
#from Reading_dictionnary import get_WE_fasttext
from Reading_dictionnary import get_WE_Word2Vec
from Reading_dictionnary import get_WE_Glove

def get_dictionnary_benchmarks(dictionnary_file,categories_list):
    if "Glove" in dictionnary_file:
        vector_representations=get_WE_Glove(dictionnary_file)
    else:
        vector_representations=get_WE_Word2Vec(dictionnary_file)
    """else:
        vector_representations=get_WE_fasttext(dictionnary_file)"""
    vector_dictionnary={}
    for row in categories_list:
        for i in range(len(row[1])):
            for key,val in vector_representations.items():
                if key==row[1][i]:
                    vector_dictionnary.update({key:val})
    return vector_dictionnary

def get_true_clusters(labeled_data_directory,dictionnary):
    categories=get_clustering_categories(labeled_data_directory)
    categories_representation=get_dictionnary_benchmarks(dictionnary,categories)
    true_word_clusters=[]
    i=0
    for row in categories:
        for word in categories_representation.keys():
            if word in row[1] and word not in [true_word_clusters[i][1] for i in range(len(true_word_clusters))]: 
                true_word_clusters.append([row[0],word,i])
        i+=1
    return [true_word_clusters[i][2] for i in range(len(true_word_clusters))]

def get_kmeans_predicted_clusters(word_representions):
    Num_clusters=9
    #from dictionnary type to transposed dataframe
    Y=pd.DataFrame(data=word_representions).T
    X=Y.values
    #Clustering the data using sklearn library
    kclusterer = KMeansClusterer(Num_clusters, distance=nltk.cluster.util.euclidean_distance, repeats=25, avoid_empty_clusters=False)
    predicted_clusters= kclusterer.cluster(X, assign_clusters=True, )
    return predicted_clusters

def get_MeanShift_clusers(categories_representation):
    #from dictionnary type to transposed dataframe
    Y=pd.DataFrame(data=categories_representation).T
    X=Y.values
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=9000)
    ms=MeanShift(bandwidth=bandwidth,bin_seeding=True)
    ms.fit(X)
    predicted_clusters=ms.predict(X)
    return predicted_clusters

def get_Agglometrative_clusters(categories_representation):
    Num_clusters=9
    Y=pd.DataFrame(data=categories_representation).T
    X=Y.values
    Agg=AgglomerativeClustering(n_clusters=Num_clusters).fit(X)
    predicted_clusters=Agg.fit_predict(X)
    return predicted_clusters

def get_DBSCAN_clusters(categories_representation):
    #from dictionnary type to transposed dataframe
    Y=pd.DataFrame(data=categories_representation).T
    X=Y.values
    B=DBSCAN(eps=0.5,min_samples=5).fit(X)
    predicted_clusters=B.fit_predict(X)
    return predicted_clusters

def get_predicted_clusters(K_mean,Mean_shift,DB_SCAN,Agglometrative,labeled_data_directory,dictionnary):
    categories=get_clustering_categories(labeled_data_directory)
    categories_representation=get_dictionnary_benchmarks(dictionnary,categories)
    if K_mean:
        predicted_clusters=get_kmeans_predicted_clusters(categories_representation)
    elif Mean_shift:
        predicted_clusters=get_MeanShift_clusers(categories_representation)
    elif AgglomerativeClustering:
        predicted_clusters=get_Agglometrative_clusters(categories_representation)
    else:
        predicted_clusters=get_DBSCAN_clusters()
    return predicted_clusters
