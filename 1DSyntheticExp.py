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
import seaborn as sns
import scipy.stats
from scipy.stats.kde import gaussian_kde

from sklearn.neighbors.kde import KernelDensity
# generate two classes with different means but the same distribution
# estimate condtion probability functions: monotonic function and non-monotonic function. The prob function is for class 1 and the class 2 is 1- prob
'''
mu_1, sigma_1 = 2, 0.5
mu_2, sigma_2 = -2, 0.5
#s1 = np.random.normal(mu_1, sigma_1, 500)
s1 =5- scipy.stats.expon.rvs(size=1000)
#s2 = np.random.normal(mu_2, sigma_2, 500)
s2 =-5 +scipy.stats.expon.rvs(size=1000)
s1 = np.sort(s1)
s2 = np.sort(s2)
s1 = np.reshape(s1,(-1,1))
s2 = np.reshape(s2,(-1,1))
y1 = KernelDensity(kernel='exponential',bandwidth=0.75).fit(s1)#p(x\y=1)
y2 = KernelDensity(kernel='exponential',bandwidth=0.75).fit(s2)
s = np.concatenate((s2,s1),axis=0)# p(y=1)=1/2

y = KernelDensity(kernel='exponential').fit(s)# p(x)
start = np.min(s)
end = np.max(s)
print(y1)
print((y))
#start = np.percentile(s,1)
#end = np.percentile(s,99)
est_len = 4000
x = np.linspace(start,end,est_len)
x = np.reshape(x,(-1,1))
log_dens1 = y2.score_samples(x)
log_dens =  y.score_samples(x)
y_pdf = log_dens1*0.5/log_dens
sns.distplot(s1, hist = False, kde = False, rug = True,
             color = 'darkblue',
             kde_kws={'linewidth': 3},
             rug_kws={'color': 'green'})
sns.distplot(s2, hist = False, kde = False, rug = True,
             color = 'darkblue',
             kde_kws={'linewidth': 3},
             rug_kws={'color': 'red'})
plt.plot(x, y_pdf)
plt.show()
'''
'''
sns.distplot(3-r, hist = False, kde = False, rug = True,
             color = 'darkblue',
             kde_kws={'linewidth': 3},
             rug_kws={'color': 'green'})

plt.show()'''

import math

def sigmoid(inputs):
    sigmoid_scores = [1 / float(1 + np.exp(- 0.5*x)) for x in inputs]
    return sigmoid_scores
'''
x_plot = (np.linspace(-15, 15, 1000));
#x = np.reshape(x,(-1, 1))
y_plot = sigmoid(x_plot)
print(np.max(y_plot))
print(np.min(y_plot))
plt.plot(x_plot,y_plot)
plt.show()'''

def softmax(inputs):

    return np.exp(inputs) / float(sum(np.exp(inputs)))

def generate_samples_conditional(ranges,label,size,mode):
    # x_plot is the samples generation range, y_plot is the conditonal prob for the class.
    #make sure that the sample size is the same as the plot size(for easy mode)
    samples = np.zeros((size))
    labels = np.zeros((size))
    print(samples.shape)
    if mode==0:
        # conditional prob is the sigmoid function

        x_plot = (np.linspace(ranges[0],ranges[1], 2*size));

        y_plot = sigmoid(x_plot)
        for i in range(size):
            dice_temp = np.random.random_sample()
            for j in range(2*size):
                if dice_temp<=y_plot[j]:
                    start = j
                    break

            index = np.random.randint(low=start,high=2*size,size=1)
            samples[i] = x_plot[index]
            labels[i] = label

        return samples,labels


samples1, labels1= generate_samples_conditional([-15,15],label=1,size=500,mode =0)
samples2 = -1*samples1
X_plot = (np.linspace(-15,15, 2000));

Y_plot = sigmoid(X_plot)
sns.distplot(samples1, hist = False, kde = False, rug = True,
             color = 'darkblue',
             kde_kws={'linewidth': 3},
             rug_kws={'color': 'green'})
sns.distplot(samples2, hist = False, kde = False, rug = True,
             color = 'darkblue',
             kde_kws={'linewidth': 3},
             rug_kws={'color': 'red'})
plt.plot(X_plot, Y_plot)
plt.show()

# using L2-SVM and V-matrix to estimate the conditional probability

