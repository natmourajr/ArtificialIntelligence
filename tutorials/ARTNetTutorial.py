from ART import ARTNet as ARTLib

import numpy as np

import time

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


iris = datasets.load_iris()
X = iris.data
Y = iris.target


print "Creating Train Parameters Obj"
trn_params = ARTLib.TrainParameters()
trn_params.itrn = [1,2,3]
trn_params.itst = [4]
trn_params.Show()


print "Creating a ARTNet Obj"
art = ARTLib.ART(4,3, dissimilarity_func='euc', rho=0.001, trn_params=trn_params, show=False)


print "Creating a new neuron"
#art.Show()
art.CreateNeuron(X[0,:].T)
#art.Show()

art.CreateNeuron(X[1,:].T)
#art.Show()

art.CreateNeuron(X[2,:].T)
art.Show()

art.CreateNeuron(X[3,:].T)

print "Checking input"
[neuron, dissimilarity] = art.CloserNeuron(X[2,:].T)

print "X[2,:]: ", X[2,:].T
print "closer neuron: ", art.activate_neurons[neuron,:]
print "smaller diss: ", dissimilarity

print "Forgetting a Neuron"
art.ForgetNeuron(0)
art.Show()

print "Forgetting the second neuron"
art.ForgetNeuron(0)
art.Show()

print "Reactivating the first forget neuron"
art.ReactivateNeuron(1)
art.Show()

print "CheckInputs in Forget Neuron"
[neuron, dissimilarity] = art.CloserForgotNeuron(X[0,:].T)
print "X[0,:]: ", X[0,:].T
print "closer neuron: ", art.forgot_neurons[neuron,:]
print "smaller diss: ", dissimilarity

print "Update Neuron" 
[neuron, dissimilarity] = art.CloserNeuron(X[4,:].T)
art.UpdateNeuron(neuron,X[4,:].T)

art.Show()