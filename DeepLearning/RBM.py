import os
import sys
import timeit

import numpy
import random

import theano

class RBM(object):
	def __init__(self, rng, n_in, n_hidden, W=None, hbias=None, vbias=None, show=False):
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
		
		