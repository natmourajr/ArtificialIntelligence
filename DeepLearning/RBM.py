import os
import sys
import timeit

import numpy
import random

import theano
from theano.tensor.shared_randomstreams import RandomStreams

class RBM(object):
	def __init__(self, rng, n_in, n_hidden, theano_rng=None, W=None, hbias=None, vbias=None, show=False):
		if show:
			print "RBM Constructor"
		
		self.rng = rng	
		self.n_in = n_in
		self.n_hidden = n_hidden
		
		# Initializing Weights
		
		# `W` is initialized with `W_values` which is uniformely sampled
		# from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		if W is None:
			W_values = numpy.asarray(rng.uniform(
			low=-numpy.sqrt(6. / (n_in + n_hidden)),
			high=numpy.sqrt(6. / (n_in + n_hidden)),
			size=(n_in, n_hidden)),
			dtype=theano.config.floatX)
			
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		self.W_init = W_values
		
		# 'hbias' is initialized with 'hbias_values' which is a zero-vector		
		if hbias is None:
			hbias_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
			hbias = theano.shared(value=hbias_values, name='hbias', borrow=True)
		self.hbias = hbias
		
		# 'vbias' is initialized with 'vbias_values' which is a zero-vector		
		if vbias is None:
			vbias_values = numpy.zeros((n_in,), dtype=theano.config.floatX)
			vbias = theano.shared(value=hbias_values, name='vbias', borrow=True)
		self.vbias = vbias
		
		self.params = [self.W, self.hbias, self.vbias]
		
		if theano_rng is None:
			theano_rng = RandomStreams(rng.randint(2 ** 30))
	
	def free_energy(self, visible_sample):
		''' Function to compute the free energy '''
		wx_b = T.dot(visible_sample, self.W) + self.hbias
		vbias_term = T.dot(visible_sample, self.vbias)
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term
		
	def propagate_up(self, visible):
		'''This function propagates the visible units activation upwards to the hidden units
		Note that we return also the pre-sigmoid activation of the 
		layer. As it will turn out later, due to how Theano deals with 
		optimizations, this symbolic variable will be needed to write 
		down a more stable computational graph (see details in the
		reconstruction cost function)'''
		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
		
	def propagate_down(self, hidden):
		'''This function propagates the hidden units activation downwards to the visible units
		Note that we return also the pre_sigmoid_activation of the
		layer. As it will turn out later, due to how Theano deals with
		optimizations, this symbolic variable will be needed to write
		down a more stable computational graph (see details in the
		reconstruction cost function)'''
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
		