#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:45:21 2020

@author: dongshuai
"""

from ClassificationNet import ClassificationNet
import argparse

from common.datasets import load_data_conv
#from collections import Counter
import numpy as np
from collections import Counter

import warnings
warnings.filterwarnings("ignore")
        
#0	T恤（T-shirt/top）
#1	裤子（Trouser）
#2	套头衫（Pullover）
#3	连衣裙（Dress）
#4	外套（Coat）
#5	凉鞋（Sandal）
#6	衬衫（Shirt）
#7	运动鞋（Sneaker）
#8	包（Bag）
#9	靴子（Ankle boot）
def symmetric_noise(y,noise_percent,k=10):
    y_noise = y.copy()
    
    indices = np.random.permutation(len(y))
    for i, idx in enumerate(indices):
        if i < noise_percent * len(y):
            y_noise[idx] = np.random.randint(k, dtype=np.int32)
    return y_noise

def asymmetric_noise(y,noise_percent,k=10):
    y_noise = y.copy()
    for i in range(k):
        indices = np.where(y == i)[0]
        for j, idx in enumerate(indices):
            if j < noise_percent * len(indices):
                #sandal->sneaker
                if i == 5:
                    y_noise[idx] = 7
                #pullover->shirt
                elif i == 2:
                    y_noise[idx] = 6
                #coat->pullover
                elif i == 4:
                    y_noise[idx] = 2
                #sneaker->sandal
                elif i == 7:
                    y_noise[idx] = 5
                #shirt->t-pullover
                elif i == 6:
                    y_noise[idx] = 2
                #T-shirt/top->Dress
                elif i == 0:
                    y_noise[idx] = 3
                #Dress->Coat
                elif i == 3:
                    y_noise[idx] = 4
    return y_noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fashion-mnist', choices=['mnist', 'fashion-mnist', 'svhn'])   
    parser.add_argument('--dataset_channel',default=1,type=int,choices=[1,3])
    parser.add_argument('--classifier_train_epochs', default=20, type=int)
    parser.add_argument('--classifier_smp_start_epoch', default=5, type=int)
    parser.add_argument('--classifier_train_batch_size', default=256, type=int)
    parser.add_argument('--classifier_model_path', default='./classifier_%s_c%s.model')
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--noise_percent', default=0.5, type=float)
    parser.add_argument('--y_pseudo_path', default='y_pseudo_asymmetric_noise')
    parser.add_argument('--y_pseudo_generate', default=False)
    args = parser.parse_args()
    
    x,y = load_data_conv(args.dataset)

    if args.y_pseudo_generate:
        y_pseudo = asymmetric_noise(y,args.noise_percent)
        np.save(args.y_pseudo_path,y_pseudo)
    else:
        y_pseudo = np.load(args.y_pseudo_path+'.npy')
    
    print(Counter(y_pseudo))
    correct = np.sum(y_pseudo==y)
    acc = correct/len(y)
    print('#'*20,'original y_pseudo acc:%f'%acc)
    
    classifier = ClassificationNet(classes=args.n_clusters,channel=args.dataset_channel)

    classifier.fit(x,y_pseudo,y,num_epochs=args.classifier_train_epochs,start_epoch=args.classifier_smp_start_epoch,batch_size=args.classifier_train_batch_size)
    classifier.save_model(args.classifier_model_path%(args.dataset,str(args.dataset_channel)))
    
#    classifier.load_model(args.classifier_model_path%(args.dataset,str(args.dataset_channel)))
    
    y_pred,_,_ = classifier.predict(x)
          
    correct = np.sum(y_pred==y)
    acc = correct/len(y)
    print('#'*20,'final acc:%.3f'%acc)

