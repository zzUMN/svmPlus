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
        y_av = 0
        for i in range(n_samples):
            if y[i]==1:
                y_av = y_av+1

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
                            for j in range(n_samples - i):
                                if X_temp[i + j] == v_pivot:
                                    num_fre = num_fre + 1
                        else:
                            if X_temp[i]<v_pivot:
                                v_pivot = X_temp[i]
                                for j in range(n_samples-i):
                                    if X_temp[i+j]==v_pivot:
                                        num_fre = num_fre + 1

                        delta_temp[i] = float(float(num_fre)/float(n_samples))

                    theta_temp = np.zeros((n_samples, 1))
                    for i in range(n_samples):
                        if i == 0:
                            theta_temp[i] = delta_temp[i]
                        else:
                            theta_temp[i] = theta_temp[i-1] + delta_temp[i]

                    V_hat = theta_temp

                    for i in range(n_samples):
                        V[X_temp_ind[i], X_temp_ind[i:n_samples]] = V[X_temp_ind[i], X_temp_ind[i:n_samples]]*V_hat[i]
                        V[X_temp_ind[i+1:n_samples], X_temp_ind[i]] = V[X_temp_ind[i+1:n_samples], X_temp_ind[i]]*V_hat[i]



            else:

                for k in range(n_features):
                    X_temp_ind = np.argsort(X[:,k]) # ascending order
                    X_temp_ind2 = np.argsort(-X[:, k])# descending order
                    X_temp1 = np.sort(X[:, k])
                    X_temp2 = np.sort(-X[:,k])
                    X_temp2 = -X_temp2
                    delta_temp = np.zeros((n_samples, 1))
                    num_fre = 0
                    for i in range(n_samples):
                        if i ==0: # find next and equal values.
                            num_fre = 0
                            v_pivot = X_temp1[i]
                            for j in range(n_samples - i):
                                if X_temp1[i + j] == v_pivot:
                                    delta_temp[i] = delta_temp[i]+float(float(1) /float(y_av))*y[X_temp_ind[i + j]]*1
                        else:
                            if X_temp1[i]>v_pivot:
                                v_pivot = X_temp1[i]
                                for j in range(n_samples-i):
                                    if X_temp1[i+j]==v_pivot:
                                        delta_temp[i] = delta_temp[i]+float(float(1)/float(y_av))*y[X_temp_ind[i+j]]*1

                    theta_temp = np.zeros((n_samples, 1))
                    for i in range(n_samples):
                        if i == 0:
                            theta_temp[i] = delta_temp[i]
                        else:
                            theta_temp[i] = theta_temp[i - 1] + delta_temp[i]

                    V_hat = np.zeros((n_samples,1)) # start update the V_hat with the ascending order
                    for i in range(n_samples):
                        if i ==0:
                            x_pivot = X_temp2[i]
                            for j in range(n_samples-i-1):
                                if X_temp2[i+j] == x_pivot:
                                    V_hat[i] =V_hat[i]+float(float(1)/float(n_samples))/float(delta_temp[n_samples-(i+j)-1](1-delta_temp[n_samples-(i+j)-1])+0.0000001)
                        else:
                            V_hat[i] = V_hat[i-1]
                            if X_temp2[i]< x_pivot:
                                x_pivot = X_temp2[i]
                                for j in range(n_samples-i-1):
                                    if X_temp2[i+j] == x_pivot:
                                        V_hat[i] = V_hat[i] + float(float(1)/float(n_samples))/float(delta_temp[n_samples-(i+j)-1](1-delta_temp[n_samples-(i+j)-1])+0.0000001)

                    for i in range(n_samples):
                        V[X_temp_ind[i], X_temp_ind[i:n_samples]] = V[X_temp_ind[i], X_temp_ind[i:n_samples]] * V_hat[i]
                        V[X_temp_ind[i + 1:n_samples], X_temp_ind[i]] = V[X_temp_ind[i + 1:n_samples], X_temp_ind[i]] * V_hat[i]

        V = V / np.amax(V)

        for i in range(n_samples):
            max_temp = V[i, i]
            # print('Diagonal elements: ')
            # print(max_temp)
            if (np.amax(V[:, i]) > max_temp) | (np.amax(V[i, :]) > max_temp):
                V[i, :] = 0 + 0.0000001
                V[:, i] = 0 + 0.0000001
                V[i.i] = max_temp

        # V = np.log(V+np.ones((n_samples, n_samples))) + 0.000001*np.eye(n_samples)

        print('V-matrix non zero :')
        print(np.transpose(np.nonzero(V)))
        print('V-matrix min: ')
        print(np.min(V))
        print('V_matrix max: ')
        print(np.max(V))

        c_V = np.linalg.cond(V)
        # c_V = np.linalg.norm(V)*np.linalg.norm(np.linalg.pinv(V))
        print('Vmatrix condition number is :' + str(c_V))
        # print(c_V)
        return V




