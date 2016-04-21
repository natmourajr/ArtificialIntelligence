from ART import ARTNet as ARTLib

import numpy as np

import time

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


iris = datasets.load_iris()
X = iris.data
Y = iris.target

print "creating a ARTNet Obj"
art = ARTLib.ART(10,show=False)

art.Show()