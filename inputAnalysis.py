import numpy as np
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA

from featureExtract import FeatureExtract
from synth_group import synthGroup
from split_data import Split
from sklearn import datasets
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from split_data import Split
from sklearn.manifold import TSNE
n_sne = 7000
from sklearn.tree import DecisionTreeClassifier        #Decision Tree
from sklearn.ensemble import RandomForestClassifier    #Random Forest
from sklearn.neural_network import MLPClassifier       #Neural Network
from sklearn.svm import SVC                            #SVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_CLASSES = 2

# analyze the breast cancer input data
features = datasets.load_breast_cancer().data
names =datasets.load_breast_cancer().feature_names
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

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(train_features)
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=train_labels)

from sklearn.decomposition import TruncatedSVD

train_reduced = TruncatedSVD(n_components=20, random_state=0).fit_transform(train_features)

train_embedded = TSNE(n_components=2, perplexity=15, verbose=2).fit_transform(train_reduced)
#fig = plt.figure(figsize=(10,10))
#ax = plt.axes(frameon=False)
#plt.setup(ax, xticks=(), yticks=())
plt.subplot(122)
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace = 0.0, hspace=0.0)
plt.scatter(train_embedded[:, 0], train_embedded[:, 1], c=train_labels, marker="x")
plt.savefig('tsneEmbedTest.pdf')

# analyze the MNIST input features

from synth_group import synthGroup
from split_data import Split
mnist_image_height = 28
mnist_image_width = 28

# generate the synthetic dataset based on the MNIST dataset
syn_datasets = synthGroup(2000, 2, 1) # 50 samples, 2 classes(normal, abnormal), unbalance ratio
labels_class1 = [0,1,5]
labels_class2 = [5,8,9]
syn_class1, syn_group1 = syn_datasets.generate_group(1000, labels_class1, 1)
syn_class2, syn_group2 = syn_datasets.generate_group(1000, labels_class2, 2)
train_features = np.zeros((2000, 784))
train_labels = np.zeros((2000, 1))

for i in range(2000):
    if i < 1000:
        train_features[i, :] = np.reshape(syn_class1[i, :, :], (784, ))
        train_labels[i] = -1
    else:
        train_features[i, :] = np.reshape(syn_class2[i-1000, :, :], (784, ))
        train_labels[i] = 1

train_features = (train_features-np.min(train_features))/(np.max(train_features)-np.min(train_features))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(train_features)
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=train_labels)

from sklearn.decomposition import TruncatedSVD

train_reduced = TruncatedSVD(n_components=20, random_state=0).fit_transform(train_features)

train_embedded = TSNE(n_components=2, perplexity=15, verbose=2).fit_transform(train_reduced)
#fig = plt.figure(figsize=(10,10))
#ax = plt.axes(frameon=False)
#plt.setup(ax, xticks=(), yticks=())
plt.subplot(122)
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace = 0.0, hspace=0.0)
plt.scatter(train_embedded[:, 0], train_embedded[:, 1], c=train_labels, marker="x")
plt.savefig('tsneEmbedTestMN.pdf')
