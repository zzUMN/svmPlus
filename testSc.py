import numpy as np
import cvxopt.solvers
import logging
from kernel import Kernel
from Vmatrix import Vmatrix
from svmModified import SVMPredictor, SVMTrainer
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools


num_samples = 100
num_features = 2
grid_size = 20
samples = np.matrix(np.random.normal(size=num_samples*num_features).reshape(num_samples, num_features))
x_min, x_max = samples[:,0].min()-1, samples[:,0].max()+1
y_min, y_max = samples[:,1].min()-1, samples[:,1].max()+1

labels = 2*(samples.sum(axis=1)>0)-1
for mode in range(1,4):
    trainer = SVMTrainer(Kernel.linear(), 0.1)
    predictor = trainer.train(samples, labels, mode=mode)
    grid_size = 20

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max,grid_size), indexing='ij')
    flatten = lambda m : np.array(m).reshape(-1)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1,2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=cm.Paired, levels = [-0.001, 0.001], extend = 'both', alpha =0.8)

    plt.scatter(flatten(samples[:, 0]), flatten(samples[:, 1]), c = flatten(labels), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    filename = "svmtest"+str(mode)+".pdf"
    plt.savefig(filename)