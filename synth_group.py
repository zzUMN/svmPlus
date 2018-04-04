from __future__ import print_function
import random
from os import listdir
import glob

import numpy as np
from scipy import misc
import tensorflow as tf
import h5py

from keras.datasets import mnist
from keras.utils import np_utils

import matplotlib.pyplot as plt

#Setting the random seed so that the results are reproducible.
random.seed(101)

#Setting variables for MNIST image dimensions
mnist_image_height = 8
mnist_image_width = 8
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
digits = load_digits()
X = digits["data"]
y = digits["target"]
#X = preprocessing.scale(X)
n_samples = X.shape[0]
X_new = np.zeros((n_samples,8,8))
for i in range(n_samples):
    X_new[i, :, :] = np.reshape(X[i, :] ,(8 ,8))

#kf = KFold(n_splits=5)
X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.2, stratify=y)

#Import MNIST data from keras
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
#print(X_train.shape)
#print(y_train.shape) # load data from sklearn
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

class synthGroup(object):
    def __init__(self, num_sample, num_classes, ratio):# the total number of samples, the number of the classes, the unbalanced ratio between classes
        self.num_sample = num_sample
        self.num_classes = num_classes
        self.ratio = ratio

    def seperate_digits(self,samples, y, labels, label):
        num_samples = samples.shape[0]
        synth_indices = []

        for j in range(num_samples):
            if y[j] == label:
                    synth_indices.append(j)
        num_target = len(synth_indices)
        indice_chosen = np.random.randint(0,num_target,1)
        synth_indices = np.array(synth_indices)
        return synth_indices[indice_chosen]

    def generate_group(self,num_samples, labels, c): # the number of this group, the labels of the original digit images, the class number of the newly generated group
        synth_labels = []

        # Define synthetic data
        synth_data = np.ndarray(shape=(num_samples, mnist_image_height, mnist_image_width),
                                dtype=np.float32)
        num_labels = len(labels)
        digit_indices= np.random.randint(0,num_labels, num_samples)

        num_digits = []
        for i in range(num_samples):

            indice_temp = digit_indices[i]
            num_digits.append(labels[indice_temp])

        for j in range(num_samples):
            label_temp = num_digits[j]
            synth_indice = self.seperate_digits(X_train, y_train, labels ,label_temp)
            synth_data[j, :, :] = X_train[synth_indice, :, :]
            synth_labels.append(c)

        return synth_data, synth_labels


