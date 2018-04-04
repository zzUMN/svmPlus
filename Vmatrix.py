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
            for i in xrange(n_samples):
                for j in xrange(n_samples):
                    for k in xrange(n_features):
                        if X[i, k] > X[j, k]:
                            maxTemp = X[i, k]
                        else:
                            maxTemp = X[j, k]
                        V[i, j] = V[i, j]* (np.amax(X) - maxTemp)

        else:
            if mode == 2: # Mu can be estimated according to the training data
                theta = 1
                for i in range(n_samples):
                    for j in range(n_samples):
                        for k in range(n_features):
                            freq = np.where((X[:, k] > np.maximum(X[i, k], X[j, k])))
                            freq = np.asarray(freq)
                            freqNum =(freq.shape)
                            if freqNum[1]==0:
                                V[i,j] = 0
                                break

                            V[i, j] = V[i, j]*(freqNum[1]/n_samples)

            else: # mode 3 is the general and stable edition
                X_new = X
                for t in range (n_samples):
                    if(y[t]==0):
                        X_new[t, :] = 0

                for i in range(n_samples):
                    for j in range(n_samples):
                        for k in range(n_features):
                            y_pos = sum(y)# y = {0,1}

                            freq = np.where(((X_new[:, k]) > np.maximum(X[i, k], X[j, k])))
                            freq = np.asarray(freq)
                            freqNum = (freq.shape)
                            F_star = freqNum[1]/y_pos
                            theta = 1/(F_star*(1-F_star)+0.00001)

                            freq_to = np.where((X[:, k] > np.maximum(X[i, k], X[j, k])))
                            freq_to = np.asarray(freq_to)
                            freqNum_to = (freq_to.shape)
                            mu = freqNum_to[1]/n_samples

                            V[i, j] = V[i, j] * mu * theta

        # double check the Vmatrix to avoid the ill-condition
        for i in range(n_samples):
            max_temp = V[i,i]
            if (np.amax(V[:,i]) > max_temp) | (np.amax(V[i,:]) > max_temp):
                V[i, :] = 0
                V[:, i] = 0
                V[i. i] = max_temp


        c_V = np.linalg.cond(V)
        #c_V = np.linalg.norm(V)*np.linalg.norm(np.linalg.pinv(V))
        print('Vmatrix condition number is :' +str(c_V))
        #print(c_V)
        return V, theta


