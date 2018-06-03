from __future__ import print_function
import random
from os import listdir
import glob
from sklearn.decomposition import PCA
import numpy as np
from scipy import misc
import tensorflow as tf
import h5py
import matplotlib as mp
mp.use('TkAgg')
from kernel import Kernel
from Vmatrix import Vmatrix
from svmModified import SVMPredictor, SVMTrainer
#from keras.datasets import mnist
#from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from synth_group import synthGroup
from split_data import Split
mnist_image_height = 28
mnist_image_width = 28

# generate the synthetic dataset based on the MNIST dataset
syn_datasets = synthGroup(100, 2, 1) # 50 samples, 2 classes(normal, abnormal), unbalance ratio
labels_class1 = [5,8,9]
labels_class2 = [2,3,5]
overlap = 5
len1 = 50
len2 = 50
syn_class1, syn_group1, sub_label1 = syn_datasets.generate_group(len1, labels_class1, 1)
syn_class2, syn_group2, sub_label2 = syn_datasets.generate_group(len2, labels_class2, 2)
train_features = np.zeros((100, 64))
train_labels = np.zeros((100, 1))
#for i in range(2000):# resize

for i in range(100):
    if i < len1:
        train_features[i, :] = np.reshape(syn_class1[i, :, :], (64, ))
        train_labels[i] = -1
    else:
        train_features[i, :] = np.reshape(syn_class2[i-len1, :, :], (64, ))
        train_labels[i] = 1
train_subgroup = np.concatenate((sub_label1,sub_label2), axis=0)
train_features = (train_features-np.min(train_features))/(np.max(train_features)-np.min(train_features))
print(train_features)
print(train_features.shape)
print(train_labels.shape)

# The umbalance mode step 1, use the less samples group to generate balance dataset.
ratio = len1/len2
print("The umbalance ratio :")
print(ratio)
res = len1-ratio*len2
print("The remains : ")
print(res)
train_features_less = train_features[len1:len1+len2-1,:]
train_labels_less = train_labels[len1:len1+len2-1]
train_subgroup_less = train_subgroup[len1:len1+len2-1]
for i in range(ratio-1):
    train_features = np.concatenate((train_features,train_features_less),axis=0)
    train_labels = np.concatenate((train_labels,train_labels_less),axis=0)
    train_subgroup = np.concatenate((train_subgroup,train_subgroup_less),axis=0)

train_features = np.concatenate((train_features,train_features_less[len1:len1+res-1,:]),axis=0)
train_labels = np.concatenate((train_labels,train_labels_less[len1:len1+res-1]),axis=0)
train_subgroup = np.concatenate((train_subgroup,train_subgroup_less[len1:len1+res-1]),axis=0)

#pca = PCA(n_components= 625)
#pca.fit(train_features)
        #componets = pca.components_
        #convariance = pca.explained_variance_

        #pcaPara = np.concatenate((componets,convariance),axis = 1)
#train_features= pca.fit_transform(train_features)

syn_datasets1 = synthGroup(100, 2, 1) # 50 samples, 2 classes(normal, abnormal), unbalance ratio
syn_class1, syn_group1, sub_labelt1 = syn_datasets1.generate_group(50, labels_class1, 1)
syn_class2, syn_group2, sub_labelt2 = syn_datasets1.generate_group(50, labels_class2, 2)
test_features = np.zeros((100, 64))
test_labels = np.zeros((100, 1))

for i in range(100):
    if i < 50:
        test_features[i, :] = np.reshape(syn_class1[i, :, :], (64, ))
        test_labels[i] = -1
    else:
        test_features[i, :] = np.reshape(syn_class2[i-50, :, :], (64, ))
        test_labels[i] = 1
test_subgroup = np.concatenate((sub_labelt1,sub_labelt2), axis=0)
test_features = (test_features-np.min(test_features))/(np.max(test_features)-np.min(test_features))
#pca = PCA(n_components= 625)
#pca.fit(test_features)
        #componets = pca.components_
        #convariance = pca.explained_variance_
#test_features= pca.fit_transform(test_features)
#train_features = StandardScaler().fit_transform(train_features)
# experiment comparison
c = [0.000001,0.000002,0.000005,0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
para_select = len(c)
for mode in range(2,4):
    '''if mode == 3:
        for n in range(100):
            if train_labels[n]==-1:
                train_labels[n] = 0
        for m in range(100):
            if test_labels[m]==-1:
                test_labels[m] = 0'''
    error_select = np.inf
    #c_select = np.inf
    for s in range(para_select):
        trainer_temp = SVMTrainer(Kernel.linear(), c[s])
        result_temp = []
        error_temp = []
        split_data = Split()
        for f in range(5):# 5-fold to choose the optimal parameter
            # contruct cross validation with 5-fold
            #trainf_features, valid_features, trainf_labels, valid_labels = train_test_split(train_features, train_labels,test_size=0.2, stratify=train_labels)

            trainf_features, validf_features, trainf_labels, validf_labels, size_train, size_valid = split_data.K_fold(train_features, train_labels, 5, f+1)
            predictor_temp = trainer_temp.train(trainf_features, trainf_labels, mode=mode)
            #size_valid = len(validf_labels)



            for i in range(size_valid):
                point = validf_features[i, :]
                result_temp.append(predictor_temp.predict(point))
                error_temp.append(np.abs(validf_labels[i] - predictor_temp.predict(point)))

        error_score = sum(error_temp)/(2*5*size_valid)
        print(error_score)
        if (error_score < error_select):
            error_select = error_score
            c_select = c[s]

    print(error_select)
    print(c_select)
    trainer = SVMTrainer(Kernel.linear(), c_select)
    predictor = trainer.train(train_features, train_labels, mode=mode)

    #error = predictor.score(test_features, test_labels)
    error_num = 0
    size_test = len(test_labels)
    error_test = []
    result_test = []
    for i in range(size_test):
        point = test_features[i, :]
        result_test.append(predictor.predict(point))

        error_test.append(np.abs(test_labels[i] - predictor.predict(point)))
    print("The accuracy of mode " + str(mode) + " is " + str(1-sum(error_test)/(2*size_test)))
    print(error_test)
    print(result_test)
    print(test_labels)

# plot the density of different classes:
    #predictor.display_Density(train_features,train_labels,'trainResult.pdf')
    #predictor.display_Density(test_features, test_labels,'testResult.pdf')

# monitor the performance of overlap digits:
    print('train_subgroup: ')
    print(train_subgroup.shape)
    predictor.subgroup_monitor(train_features,train_labels,train_subgroup,overlap=overlap,mode=0)
    predictor.subgroup_monitor(train_features,train_labels,train_subgroup,overlap=overlap,mode=1)

    predictor.subgroup_monitor(test_features,test_labels,test_subgroup,overlap=overlap,mode=0)
    predictor.subgroup_monitor(test_features,test_labels,test_subgroup,overlap=overlap,mode=1)
