from __future__ import print_function
import random
from os import listdir
import glob

import numpy as np
from scipy import misc
import tensorflow as tf
import h5py

class Vmatrix(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def calculateEle(self, X, y, mode):
        n_samples, n_features = X.shape
        if mode == 1:
            theta = 1
            dMu = 