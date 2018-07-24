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

                y_pos = 0
                for iii in range(n_samples):
                    if y[iii] == 1:
                        y_pos = y_pos + 1
                for i in range(n_samples):
                    for j in range(n_samples):

                        for k in range(n_features):

                            freq_to = np.where((X[:, k] > np.maximum(X[i, k], X[j, k])))
                            freq_to = np.asarray(freq_to)
                            freqNum_to = (freq_to.shape[1])
                            V_ijk_temp = 0
                            for ssN in range(freqNum_to):
                                X_temp = X[freq_to[0,ssN],k]
                                freq_temp = np.where(X[:,k]<X_temp)
                                freq_temp = np.asarray(freq_temp)
                                freqNum_temp = (freq_temp.shape[1])
                                mu_temp = freqNum_temp/n_samples
                                freq_reduce_temp = 0
                                for tt in range(freqNum_temp):
                                    if y[freq_temp[0,tt]]==-1:
                                        freq_reduce_temp = freq_reduce_temp+1

                                freqStar_temp = freqNum_temp - freq_reduce_temp
                                F_star = freqStar_temp / y_pos
                                theta_temp = 1 / (F_star * (1 - F_star) + 0.00001)
                                V_ijk_temp = V_ijk_temp+mu_temp*theta_temp

                            V[i,j] = V[i,j]*V_ijk_temp



                #print("V-matrix:")
                #print(V)
    # double check the Vmatrix to avoid the ill-condition
        '''
        for i in range(n_samples):
            max_temp = V[i,i]
            if (np.amax(V[:,i]) > max_temp) | (np.amax(V[i,:]) > max_temp):
                V[i, :] = 0
                V[:, i] = 0
                V[i. i] = max_temp
        '''

        c_V = np.linalg.cond(V)
        #c_V = np.linalg.norm(V)*np.linalg.norm(np.linalg.pinv(V))
        print('Vmatrix condition number is :' +str(c_V))
        #print(c_V)
        return V


