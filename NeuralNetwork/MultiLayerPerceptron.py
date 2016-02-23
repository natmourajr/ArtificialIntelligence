import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class InputLayer(object):
	""" Input Layer Class """
	
	def __init__(self, n_in, n_out, pre_function=None, show=False):

		""" 
    	Params:
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	pre_function: now NONE!!
    	desc: Pre-processing function
    	"""
    			
		if show:
			print ""
			print "Input Layer Constructor"
			print "n_in: ", n_in
			print "n_out: ", n_out
			print "pre_function: ", pre_function
			print ""

		self.n_in = n_in
		self.n_out = n_out
		self.pre_function = pre_function
		
	def Show(self):
		print ""
		print "InputLayer Object"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		print "Pre-Processing Function: ", self.pre_function
		print ""

class HiddenLayer(object):
	""" Hidden Layer Class """
	
	def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh, show=False):
		
		""" 
    	Params:
    	
    	rng: numpy.random.RandomState
    	desc: a random number generator used to initialize weights
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	activation: theano.Op or function
    	desc: Non linearity to be applied in this layer
    	"""
    	
		if show:
			print ""
			print "Hidden Layer Constructor"
			print "n_in: ", n_in
			print "n_out: ", n_out
			print "activation: ", activation
			print ""
		
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		
		
		# Initializing Weights
		
		# `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		
		if W is None:
			W_values = numpy.asarray(rng.uniform(
			low=-numpy.sqrt(6. / (n_in + n_out)),
			high=numpy.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)),
			dtype=theano.config.floatX)
			
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			
			# Note : optimal initialization of weights is dependent on the
        	#        activation function used (among other things).
        	#        For example, results presented in [Xavier10] suggest that you
        	#        should use 4 times larger initial weights for sigmoid
        	#        compared to tanh
        	#        We have no info for other function, so we use the same as
        	#        tanh.
			
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		self.W_init = W_values
		
		
		# 'b' is initialized with 'b_values' which is a zero-vector
		
		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.b = b
		
		
	
	def Show(self):
		print ""
		print "HiddenLayer Object"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		print "Activation Function: ", self.activation
		print "Initialization Function: ", "Random"
		print ""
    	
class OutputLayer(object):
	""" Output Layer Class """
	
	def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh, show=False):
		
		""" 
    	Params:
    	
    	rng: numpy.random.RandomState
    	desc: a random number generator used to initialize weights
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	activation: theano.Op or function
    	desc: Non linearity to be applied in this layer	
    	"""
    	
		if show:
			print ""
			print "Output Layer Constructor"
			print "n_in: ", n_in
			print "n_out: ", n_out
			print "activation: ", activation
			print ""

		self.n_in = n_in
		self.n_out = n_out
		self.W = W
		self.b = b
		self.activation = activation
		
		
		# Initializing Weights
		
		# `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		
		if W is None:
			W_values = numpy.asarray(rng.uniform(
			low=-numpy.sqrt(6. / (n_in + n_out)),
			high=numpy.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)),
			dtype=theano.config.floatX)
			
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			
			# Note : optimal initialization of weights is dependent on the
        	#        activation function used (among other things).
        	#        For example, results presented in [Xavier10] suggest that you
        	#        should use 4 times larger initial weights for sigmoid
        	#        compared to tanh
        	#        We have no info for other function, so we use the same as
        	#        tanh.
			
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		self.W_init = W_values
		
		
		# 'b' is initialized with 'b_values' which is a zero-vector
		
		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.b = b
		
	def Show(self):
		print ""
		print "OutputLayer Object"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		print "Activation Function: ", self.activation
		print "Initialization Function: ", "Random"
		print ""
		
		
class TrainParameters(object):
	def __init__(self,show=False):
		""" Train Parameters Class """
		
class MLP(object):
    """ Multi-Layer Perceptron Class """
    
    def __init__(self, rng, n_in, n_hidden, n_out, hidden_act=T.tanh, output_act=T.tanh, show=False):
    	
    	if show:
    		print ""
    		print 'MLP Class Constructor'
    		print "n_in: ", n_in
    		print "n_hidden: ", n_hidden
    		print "n_out: ", n_out
    		print "hidden_act: ", hidden_act 
    		print "output_act: ", output_act 
    		print ""

    	""" 
    	Params:
    	
    	rng: numpy.random.RandomState
    	desc: a random number generator used to initialize weights
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_hidden: integer or array of integer
    	desc: dimensionality of hidden layer
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	hidden_act: theano.Op or function
    	desc: Non linearity to be applied in the hidden layer
    	
    	output_act: theano.Op or function
    	desc: Non linearity to be applied in the output layer
    	"""
    	
    	self.InputLayer = InputLayer(n_in, n_in, pre_function=None, show=show)
    	
    	self.HiddenLayer = {}
    	
    	for i in range(len(n_hidden)):
    		if i == 0:
    			self.HiddenLayer[i] = HiddenLayer(rng, n_in=n_in, n_out=n_hidden[i],show=show)
    		else:
    			self.HiddenLayer[i] = HiddenLayer(rng, n_in=n_hidden[i-1], n_out=n_hidden[i],show=show)
    			
    	self.OutputLayer = OutputLayer(rng,n_in=n_hidden[-1],n_out=n_out,show=show)

    def Show(self):
    	""" This function show all relevant features of Neural Network Obj """
    
    	show_str = []
    	
    	# Title
    	print  "\nNeural Network Object "
    	self.InputLayer.Show()
    	for i in range(len(self.HiddenLayer)):
    		print "HiddenLayer["+str(i)+"]:"
    		self.HiddenLayer[i].Show()
    	self.OutputLayer.Show()
    	
    def Train(self, input=None, target=None, trn_params=None):
    	print "Train Function"
    	
    	"""
    	Params: 
    	
    	input: vector of natural numbers
    	desc: Vector with inputs
    	
    	target: vector of natural numbers
    	desc: Vector with targets
    	
    	trn_params: Object of trn_params class
    	desc: Same as TrainParameters Class
    	
    	"""
    	
    	self.input = input
    	self.target = target
    	self.trn_params = trn_params
    	
    	
    	
    	