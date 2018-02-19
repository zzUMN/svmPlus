from __future__ import print_function
import random
from os import listdir
import glob

import numpy as np
from scipy import misc
import tensorflow as tf
import h5py
from scipy.stats import norm

class Vmatrix(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def calculateEle(self, X, y, mode):
        n_samples, n_features = X.shape
        V = np.ones((n_samples,n_samples))

        if mode == 1: # the data belonged to a upper-bounded support (-inf, B]
            theta = 1
            for i in xrange(100):
                for j in xrange(100):
                    for k in xrange(2):
                        if X[i, k] > X[j, k]:
                            maxTemp = X[i, k]
                        else:
                            maxTemp = X[j, k]
                        V[i, j] = V[i, j]* (np.amax(X) - maxTemp )

        else:
            if mode == 2: # Mu can be estimated according to the training data
                theta = 1
                for i in range(n_samples):
                    for j in range(n_samples):
                        for k in range(n_features):
                            freq = np.where(np.logical_and(X[:, k] > np.amax(X[i, k], X[j, k])))
                            freqNum = freq.shape
                            V[i, j] = V[i, j]*(freqNum[1]/n_samples)

            '''else:
                theta = 
                for i in range(n_samples):
                    for j in range(n_samples):
                        for k in range(n_features):
                            if X[]'''

        return V, theta


