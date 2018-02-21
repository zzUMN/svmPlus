import argparse
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import cvxopt.solvers
import logging
from kernel import Kernel
from Vmatrix import Vmatrix
from svmModified import SVMPredictor, SVMTrainer
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from split_data import Split

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_CLASSES = 2


features = datasets.load_breast_cancer().data

# standardize the features
features = StandardScaler().fit_transform(features)

# get the number of features
num_features = features.shape[1]

# load the corresponding labels for the features
labels = datasets.load_breast_cancer().target

# transform the labels to {-1, +1}
labels[labels == 0] = -1

# split the dataset to 70/30 partition: 70% train, 30% test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size=0.3, stratify=labels)

train_size = train_features.shape[0]
test_size = test_features.shape[0]

# slice the dataset as per the batch size
train_features = train_features[:train_size - (train_size % BATCH_SIZE)]
train_labels = train_labels[:train_size - (train_size % BATCH_SIZE)]
test_features = test_features[:test_size - (test_size % BATCH_SIZE)]
test_labels = test_labels[:test_size - (test_size % BATCH_SIZE)]

c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
para_select = len(c)
for mode in range(1,4):
    error_select = np.inf
    c_select = np.inf
    for s in range(para_select):
        trainer_temp = SVMTrainer(Kernel.linear(), c[s])

        # contruct cross validation with 5-fold

        predictor_temp = trainer_temp.train(samples_train, labels_train, mode=mode)
