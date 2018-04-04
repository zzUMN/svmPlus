import numpy as np
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA

from featureExtract import FeatureExtract
from synth_group import synthGroup
from split_data import Split


mnist_image_height = 28
mnist_image_width = 28
syn_datasets = synthGroup(100, 2, 4) # 50 samples, 2 classes(normal, abnormal), unbalance ratio
labels_class1 = [1, 2, 5]
labels_class2 = [7, 8, 9]
syn_class1, syn_group1 = syn_datasets.generate_group(50, labels_class1, 1)
syn_class2, syn_group2 = syn_datasets.generate_group(50, labels_class2, 2)
train_features = np.zeros((100, 784))
#train_features = np.concatenate((syn_class1,syn_class2), axis=0)
train_labels = np.zeros((100, 1))
print(train_features.shape)
for i in range(100):
    if i < 50:
        train_features[i, :] = np.reshape(syn_class1[i, :, :], (784, ))
        train_labels[i] = -1
    else:
        train_features[i, :] = np.reshape(syn_class2[i-50, :, :], (784, ))
        train_labels[i] = 1

#train_features = (train_features-np.min(train_features))/(np.max(train_features)-np.min(train_features))

methods = ['pca']
pca = PCA(n_components= 30)
pca.fit(train_features)
        #componets = pca.components_
        #convariance = pca.explained_variance_

        #pcaPara = np.concatenate((componets,convariance),axis = 1)
pcaFeature = pca.fit_transform(train_features)
print(pcaFeature.shape)
'''
feature_generator = FeatureExtract(methods, mode=1)
features_ex = feature_generator.sequenceFeature(train_features,methods)
print(features_ex)
'''