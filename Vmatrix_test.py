import pylab as pl
import numpy as np
import matplotlib as mp
mp.use('TkAgg')
from kernel import Kernel
from Vmatrix import Vmatrix
from svmModified import SVMPredictor, SVMTrainer
import matplotlib.pyplot as plt
def gen_lin_separable_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    x1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = -1*np.ones(len(x2))

    return x1, y1, x2, y2

def gen_non_lin_separable_data():

    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]

    cov = [[1.0, 0.8], [0.8, 1.0]]

    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2,cov,50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov,50)))
    y2 = -1*np.ones(len(X2))

    return X1, y1, X2, y2

def gen_lin_separable_overlap_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    x1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = -1 * np.ones(len(x2))

    return x1, y1, x2, y2

def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    X2_train = X2[:90]
    y1_train = y1[:90]
    y2_train = y2[:90]

    X_train  = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))

    return X_train, y_train

def split_test(X1, y1, X2,y2):
    X1_test = X1[90:]
    X2_test = X2[90:]
    y1_test = y1[90:]
    y2_test = y2[90:]

    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

def plot_margin(X1_train, X2_train, clf):
    def f(x,w,b,c=0):
        return (-w[0]*x-b+c)/w[1]

    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf._support_vectors[:,0], clf._support_vectors[:,1], s=100, c="g")

    a0 = -4; a1 = f(a0,clf._weights,clf._bias,1)
    b0 =4; b1 = f(b0, clf._weights, clf._bias,1)

    pl.plot([[a0,b0],[a1,b1]], "k--")

    a0 = -4;
    a1 = f(a0, clf._weights, clf._bias,-1)
    b0 = 4;
    b1 = f(b0, clf._weights, clf._bias,-1)

    pl.plot([[a0, b0], [a1, b1]], "k--")

    pl.axis("tight")
    pl.show()

def plot_contour(X1_train, X2_train, clf):
    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    plt.scatter(clf._support_vectors[:, 0], clf._support_vectors[:, 1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))

    X = np.array([[x1,x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    print(X.shape)
    Z = np.zeros((X1.shape))
    for i in range(2500):
        #for j in range(50):
            Z[i/50, i%50] = clf.gen_conditional_prob(X[i])#.reshape(X1.shape)


    plt.contour(X1,X2,Z, [0.0], colors='k',linewidths=1, origin='lower')
    plt.contour(X1,X2,Z+1,[0.0],colors='grey',linewidths=1, origin='lower')
    plt.contour(X1,X2,Z-1,[0.0],colors='grey',linewidths=1, origin='lower')

    plt.axis("tight")
    plt.show()


def test_linear():
    X1, y1, X2, y2 = gen_lin_separable_data()
    X_train, y_train = split_train(X1,y1,X2,y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVMTrainer(Kernel.linear(),c=0)
    predictor = clf.train(X_train, y_train,mode=1)
    # y_predict = predictor.predict(X_test[0])
    y_predict = np.zeros(len(X_test))
    for i in range(len(X_test)):
        y_predict[i] = predictor.predict(X_test[i])
    correct = np.sum(y_predict==y_test)
    print("%d out of  %d predictions correct" % (correct, len(y_predict)))

    plot_margin(X_train[y_train==1], X_train[y_train==-1], predictor)

def test_non_linear():
    X1, y1, X2, y2 = gen_non_lin_separable_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVMTrainer(Kernel.gaussian(1))
    predictor = clf.train(X_train,y_train,mode=1)
    #y_predict = predictor.predict(X_test[0])
    y_predict= np.zeros(len(X_test))
    for i in range(len(X_test)):
        y_predict[i] = predictor.predict(X_test[i])

    correct = np.sum(y_predict == y_test)
    print("%d out of  %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], predictor)

def test_soft():
    X1, y1, X2, y2 = gen_lin_separable_overlap_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)
    print(y_train.shape)
    clf = SVMTrainer(Kernel.linear(), c=10)
    predictor = clf.train(X_train, y_train,mode=3)
    # y_predict = predictor.predict(X_test[0])
    y_predict = np.zeros(len(X_test))
    for i in range(len(X_test)):
        y_predict[i] = predictor.predict(X_test[i])
    correct = np.sum(y_predict == y_test)
    print("%d out of  %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], predictor)

test_soft()