#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:39:14 2020

@author: dongshuai
"""

import numpy as np

def cos_similarity(x1,x2):
    t1 = x1.dot(x2.T)
    x1_linalg = np.linalg.norm(x1,axis=1)
    x2_linalg = np.linalg.norm(x2,axis=1)
    x1_linalg = x1_linalg.reshape((x1_linalg.shape[0],1))
    x2_linalg = x2_linalg.reshape((1,x2_linalg.shape[0]))
    t2 = x1_linalg.dot(x2_linalg)
    cos = t1/t2
    
    return cos

def calculate_S(z):
    S = cos_similarity(z,z)
    return S

def calculate_rou(S,rate=0.4):
    m = S.shape[0]
    rou = np.zeros((m))
    
    t = int(rate*m*m)
    temp = np.sort(S.reshape((m*m,)))
    Sc = temp[-t]
    
    rou = np.sum(np.sign(S-Sc),axis=1) - np.sign(S.diagonal()-Sc)
    
    return rou

def get_prototype_index(S,rou,p):
    rou_max = np.max(rou)
    
    m = S.shape[0]
    ita = np.zeros((m))
    
    for i in range(m):
        if rou[i] == rou_max:
            ita[i] = np.min(S[i])
        else:
            ita[i] = S[i,i]
            for j in range(m):
                if i != j and rou[j] > rou[i]:
                    if ita[i] < S[i,j]:
                        ita[i] = S[i,j]
    return np.argsort(ita)[:p]

def get_prototypes(z,y,k,p,samples_n):
    prototypes_list = []
    for i in range(k):
        index = np.arange(z[y==i].shape[0])
        z_samples_index = np.random.choice(index,samples_n,replace=False)
        z_samples = z[y==i][z_samples_index]
        S = calculate_S(z_samples)   
        rou = calculate_rou(S)
        prototype_index = get_prototype_index(S,rou,p)
        prototypes = z_samples[prototype_index]
        prototypes_list.append(prototypes)
        
    return prototypes_list

def produce_pseudo_labels(z,y,k,p=8,samples_n=1280):
    prototypes_list = get_prototypes(z,y,k,p,samples_n)
    
    y_pseudo = y
    n = z.shape[0]
    sigma = np.zeros((n,k))
    for c in range(k):
        prototypes = prototypes_list[c]
        S = cos_similarity(z,prototypes)
        sigma[:,c] = np.mean(S,axis=1)
        
    y_pseudo = np.argmax(sigma,axis=1)
        
    return y_pseudo

