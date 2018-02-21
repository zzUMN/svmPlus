
import numpy
# x is your dataset
class Split(object):
    def split_data(self,X,y,size_train,size_valid):
        num_samples, num_features = X.shape

        training_idx = numpy.random.randint(num_samples, size=size_train)
        valid_idx = numpy.random.randint(num_samples, size=size_valid)
        training, valid = X[training_idx, :], X[valid_idx, :]
        training_labels, valid_labels = y[training_idx, :], y[valid_idx, :]

        return training, valid, training_labels, valid_labels

    def K_fold(self, X, y, k):
        num_samples, num_features = X.shape

        size_valid = num_samples/k
        size_training = num_samples-size_valid

        training, valid, training_labels, valid_labels = self.split_data(X, y, size_training, size_valid)

        return training, valid, training_labels, valid_labels
