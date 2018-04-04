import numpy as np
import cvxopt.solvers
import logging
from kernel import Kernel
from Vmatrix import Vmatrix
from svmModified import SVMPredictor, SVMTrainer
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from split_data import Split

num_samples = 100
num_features = 2
grid_size = 20
samples = np.matrix(np.random.normal(size=num_samples*num_features).reshape(num_samples, num_features))
x_min, x_max = samples[:,0].min()-1, samples[:,0].max()+1
y_min, y_max = samples[:,1].min()-1, samples[:,1].max()+1

labels = 2*(samples.sum(axis=1)>0)-1
c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
para_select = len(c)
for mode in range(3,4):
    error_select = np.inf
    c_select = np.inf
    for s in range(para_select):
        trainer_temp = SVMTrainer(Kernel.linear(), c[s])

        splitM = Split()
        samples_train, samples_valid, labels_train, labels_valid, size_valid, size_training = splitM.K_fold(samples, labels, 5, 1)

        predictor_temp = trainer_temp.train(samples_train, labels_train, mode=mode)
        grid_size = 20
        x_min, x_max = samples_valid[:, 0].min() - 1, samples_valid[:, 0].max() + 1
        y_min, y_max = samples_valid[:, 1].min() - 1, samples_valid[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max,grid_size), indexing='ij')
        flatten = lambda m : np.array(m).reshape(-1)

        result_temp = []
        '''
        for (i, j) in itertools.product(range(grid_size), range(grid_size)):
            point = np.array([xx[i, j], yy[i, j]]).reshape(1,2)
            result_temp.append(predictor_temp.predict(point))
        '''
        error_temp = []
        for i in range(size_valid):
            point = samples_valid[i,:]
            result_temp.append(predictor_temp.predict(point))
            error_temp.append(np.abs(labels_valid[i]-predictor_temp.predict(point)))

        if(sum(error_temp)<error_select):
            error_select = sum(error_temp)
            c_select = c[s]


    print(c_select)
    trainer = SVMTrainer(Kernel.linear(), c_select)
    predictor = trainer.train(samples, labels, mode=mode)
    grid_size = 20

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size), indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=cm.Paired, levels = [-0.001, 0.001], extend = 'both', alpha =0.8)

    plt.scatter(flatten(samples[:, 0]), flatten(samples[:, 1]), c = flatten(labels), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    filename = "svmtest"+str(mode)+".pdf"
    plt.savefig(filename)
    plt.close()