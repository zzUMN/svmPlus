from __future__ import print_function
import random
from os import listdir
import glob

import numpy as np
from scipy import misc
import tensorflow as tf
import h5py
from scipy.stats import norm

class V_starMatrix(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def calculateEle(self, X, y, mode):
        n_samples, n_features = X.shape
        V = np.ones((n_samples, n_samples))

        if mode == 1:  # the data belonged to a upper-bounded support (-inf, B]
            theta = 1
            for i in xrange(n_samples):
                for j in xrange(n_samples):
                    for k in xrange(n_features):
                        if X[i, k] > X[j, k]:
                            maxTemp = X[i, k]
                        else:
                            maxTemp = X[j, k]
                        V[i, j] = V[i, j] * (np.amax(X) - maxTemp)

        else:
            if mode == 2:  # Mu can be estimated according to the training data

                for k in range(n_features):
                    X_temp_ind = np.argsort(-X[:, k])
                    X_temp = np.sort(-X[:, k])
                    X_temp = -X_temp
                    delta_temp = np.zeros((n_samples, 1))
                    num_fre = 0
                    for i in range(n_samples):
                        if i ==0: # find next and equal values.
                            num_fre = 0
                            v_pivot = X_temp[i]
                        else:
                            if X_temp[i+1]<v_pivot:
                                for j in range(n_samples-i-1):
                                    if X_temp[i+j+1]==v_pivot:
                                        num_fre = num_fre + 1

                    delta_temp[i] = num_fre/n_samples

                    theta_temp = np.zeros((n_samples, 1))
                    for i in range(n_samples):
                        if i == 0:
                            theta_temp[i] = delta




