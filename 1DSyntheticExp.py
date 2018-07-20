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
from synth_group import synthGroup
from split_data import Split

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
    sigmoid_scores = [1 / float(1 + np.exp(-8*(x-0.8))) for x in inputs]
    return sigmoid_scores
'''
x_plot = (np.linspace(-15, 15, 1000));
#x = np.reshape(x,(-1, 1))
y_plot = sigmoid(x_plot)
print(np.max(y_plot))
print(np.min(y_plot))
plt.plot(x_plot,y_plot)
plt.show()'''

def sigmoid_output(inputs):
    sigmoid_scores = [1 / float(1 + np.exp(-8*(x-0.8))) for x in inputs]
    return sigmoid_scores
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
class_size = 192

samples1, labels1= generate_samples_conditional([0.5,1],label=1,size=class_size,mode =0)


samples2 = 1-1*samples1
print('sample 1 size:')
print(samples1.shape)
X_plot = (np.linspace(0,1, 2000))

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
samples1 = np.reshape(samples1,[class_size,1])
#samples1_t = np.reshape(samples1[0:100,0],[100,1])
p_con1 = sigmoid(samples1)
#for i in range(9):
 #samples1_t = np.concatenate((samples1_t,np.reshape(samples1[100*(i):100*(i+1),0], [100,1])),axis=1)

'''s = samples1[2,:]
p_con = sigmoid(s)
print('The sample  is :')
print(s)
print('the conditional prob :')
print(p_con)'''
samples2 = np.reshape(samples2, [class_size,1])
#samples2_t = np.reshape(samples2[0:100,0], [100,1])
p_con2 = sigmoid(samples2)
#for i in range(9):
 #samples2_t = np.concatenate((samples2_t,np.reshape(samples2[100*(i):100*(i+1),0],[100,1])),axis=1)



# define the feature and labels
train_features = np.concatenate((samples1, samples2),axis=0)
labels2 = labels1-2
train_labels = np.concatenate((labels1, labels2),axis=0)
test_features = train_features
test_labels = train_labels
# using L2-SVM and V-matrix to estimate the conditional probability

c = [0.000001,0.000002,0.000005,0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
para_select = len(c)
for mode in range(3,4):

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
                #print(' the error of each point')
                error_temp.append(np.abs(validf_labels[i] - (predictor_temp.predict(point))))
                #print(np.abs(validf_labels[i] - predictor_temp.predict(point)))
                '''print('the predict value:')
                print(predictor_temp.predict(point))'''

        error_score = sum(error_temp)/(size_valid)
        print('Error score :')
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
    cond_prob = []
    #test_features = np.sort(test_features)
    for i in range(size_test):
        point = test_features[i, :]
        result_test.append(predictor.predict(point))
        cond_prob.append(predictor.gen_conditional_prob(point))
        error_test.append(np.abs(test_labels[i] - (predictor.predict(point))))
    test_features = np.array(np.reshape(test_features[:,0],(2*class_size,1)))
    cond_prob = np.array(cond_prob)
    min_prob = np.min(cond_prob,axis=0)
    max_prob = np.max(cond_prob,axis=0)
    print("conditional Prob :")
    print(cond_prob)
    cond_prob = (cond_prob-min_prob)/(max_prob-min_prob)
    cond_prob = sigmoid_output(cond_prob)
    print("The accuracy of mode " + str(mode) + " is " + str(1-sum(error_test)/(size_test)))

    sns.distplot(samples1, hist=False, kde=False, rug=True,
                 color='darkblue',
                 kde_kws={'linewidth': 3},
                 rug_kws={'color': 'green'})
    sns.distplot(samples2, hist=False, kde=False, rug=True,
                 color='darkblue',
                 kde_kws={'linewidth': 3},
                 rug_kws={'color': 'red'})
    #plt.plot(X_plot, Y_plot)
    plt.plot(test_features, cond_prob,'ro')
    plt.show()



