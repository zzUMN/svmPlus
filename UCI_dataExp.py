from __future__ import print_function

import urllib
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kernel import Kernel
from Vmatrix import Vmatrix
from svmModified import SVMPredictor, SVMTrainer

df = pd.read_csv('/home/zzumn/PycharmProjects/svmPlus-master/Pima/pima_indians_diabetes.txt')

#-------------------

#defining the columns
df.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']

def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (df.apply(num_missing, axis=0),'\n') #no nans

X = np.array(df.drop(['Class'], axis = 1))
y = np.array(df['Class'])
y = 2*y-1
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size =0.8,
                                                   random_state = 7)

num_features = X_train.shape[1]# lead the prediction process make sense without knowing any infos from the test dataset
'''for i in range(num_features):
    min_temp = np.min(X_train[:,i])
    max_temp = np.max(X_train[:,i])
    X_train[:,i] = (X_train[:,i]-min_temp)/(max_temp-min_temp)
    X_test[:,i] = 0.99*(X_test[:,i]-min_temp)/(max_temp-min_temp)
'''
c = [0.000001,0.000002,0.000005,0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
para_select = len(c)
modes = [3]
for mode in modes:
    error_select = np.inf
    for s in range(para_select):
        trainer_temp = SVMTrainer(Kernel.linear(), c[s])
        result_temp = []
        error_temp = []

        for f in range(5):
            X_trainf, X_valid, y_trainf, y_valid = train_test_split(X_train, y_train, test_size=1,
                                                                random_state=7+f)

            predictor_temp = trainer_temp.train(X_trainf, y_trainf, mode=mode)
            size_valid = y_valid.shape[0]

            for i in range(size_valid):
                point = X_valid[i, :]
                result_temp.append(predictor_temp.predict(point,mod=mode))
                error_temp.append(np.abs(y_valid[i] - predictor_temp.predict(point,mod=mode)))

        error_score = sum(error_temp) / (2 * 5* size_valid)

        print(error_score)
        if (error_score < error_select):
            error_select = error_score
            c_select = c[s]

    print(error_select)
    print(c_select)
    trainer = SVMTrainer(Kernel.linear(), c_select)
    predictor = trainer.train(X_train, y_train, mode=mode)

    error_num = 0
    size_test = len(y_test)
    error_test = []
    result_test = []
    for i in range(size_test):
        point = X_test[i, :]
        result_test.append(predictor.gen_conditional_prob(point, mod=mode))

        error_test.append(np.abs(y_test[i] - predictor.predict(point,mod=mode)))
    print("The accuracy of mode " + str(mode) + " is " + str(1 - sum(error_test) / (2 * size_test)))
    print("The errors happen at")
    print(error_test)
    print("The result prediction: ")
    print(result_test)
    print("The actual labels: ")
    print(y_test)