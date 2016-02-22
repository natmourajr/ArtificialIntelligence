import os
import sys
import timeit

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
		self.W = W
		self.b = b
		self.activation = activation
		
    	
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
    	show_str = "\nNeural Network Object \n"
    	show_str = show_str +"Inputs: "+str(self.InputLayer.n_in)+"\n"
    	for i in range(len(self.HiddenLayer)):
    		show_str = show_str +"HiddenLayer["+str(i)+"]: "+str(self.HiddenLayer[i].n_out)+"\n"
    	show_str = show_str+"Outputs: "+str(self.OutputLayer.n_out)+"\n"

    	print show_str
    	
    	