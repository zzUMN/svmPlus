from __future__ import print_function
import numpy
# x is your dataset
class Split(object):
    def split_data(self,X,y, size_train, size_valid, mode, ratio):
        num_samples, num_features = X.shape
        if mode == 0:
            training_idx = numpy.random.randint(num_samples, size=size_train)
            valid_idx = numpy.random.randint(num_samples, size=size_valid)
        else :
            training_idx = (numpy.arange(0, num_samples*ratio/(ratio+1)))
            valid_idx = (numpy.arange(num_samples*ratio/(ratio+1), num_samples))
        training, valid = X[training_idx, :], X[valid_idx, :]
        training_labels, valid_labels = y[training_idx], y[valid_idx]

        return training, valid, training_labels, valid_labels

    def K_fold(self, X, y, k, group_index):
        num_samples, num_features = X.shape
        numpy.random.seed(1234)
        X = numpy.asarray(numpy.random.permutation(X))
        y = numpy.asarray(numpy.random.permutation(y))

        size_valid = num_samples/k
        size_training = num_samples-size_valid
        for i in range(group_index):
            if i ==0:
                training, valid, training_labels, valid_labels = self.split_data(X, y, size_training, size_valid, mode=1, ratio=k-1)
            else:
                X = numpy.concatenate((valid, training), axis=0)
                y = numpy.concatenate((valid_labels, training_labels), axis=0)

                training, valid, training_labels, valid_labels = self.split_data(X, y, size_training, size_valid, mode=1, ratio=k-1)

        return training, valid, training_labels, valid_labels, size_training, size_valid
