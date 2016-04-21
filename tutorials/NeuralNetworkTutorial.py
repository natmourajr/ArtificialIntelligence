from NeuralNetwork import MultiLayerPerceptron as mlpLib
import numpy as np

import time

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


iris = datasets.load_iris()
X = iris.data
Y = iris.target

target = np.zeros([X.shape[0],3])

for i in xrange(X.shape[0]):
    target[i,Y[i]] = 1

# StandardScaler
X_scaled = preprocessing.scale(X)

init_time = time.time()

mlp = mlpLib.MLP(rng=np.random.RandomState(), n_in=X.shape[1], n_hidden=[1], n_out=target.shape[1])

mlp.InputLayer.pre_function=None

trn_params = mlpLib.TrainParameters(n_epochs=100,show_freq=10)
trn_params.l1reg = 0.0
trn_params.l2reg = 0.0

trn_desc = mlp.Train(X_scaled,target,trn_params)

print "Time to train: ", time.time()-init_time, " seconds"


print "MLP Freeze: ",len(mlp.freeze)
