import os
import sys
import timeit

import numpy
import numpy as np

import random

from sklearn.cross_validation import ShuffleSplit

class TrainParameters(object):
	def __init__(self, perf_function='euc', learning_rate=0.01, perc_trn=0.5, perc_tst=0.5, show=False):
		""" Train Parameters Class 
			
			perf_function: Perfomance Function (MSE)
			
			learning_rate: Learning Rate
			
			
		"""
		self.perf_function = perf_function
		self.learning_rate = learning_rate
		
		self.perc_trn = perc_trn
		self.perc_tst = perc_tst
		
		self.itrn = None
		self.itst = None
		
class ART(object):
	""" ART Net """
	
	def __init__(self, n_in, dissimilarity_func='euc', rho=0.1, max_neurons=10, forget_thr=10, show=False):
	
		""" 
    	Params:
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	dissimilarity_func: str
    	desc: dissimilarity function
    	
    	rho: float
    	desc: Vigilance radius
    	
    	max_neurons: integer
    	desc: Max. quantity of neurons
    	
    	forget_thr: integer
    	desc: number of train epochs to forget a neuron
    	
    	"""
    	
		self.n_in = n_in
		self.dissimilarity_func = dissimilarity_func
		self.rho = rho
		self.max_neurons = max_neurons
		self.forget_thr = forget_thr
		
		self.neurons = None
		self.neurons_up = None
		
		if show:
			print ""
			print "ART Constructor"
			print "n_in: ", n_in
			print "dissimilarity_func: ",dissimilarity_func
			print "rho: ", rho
			print "max_neurons: ", max_neurons
			print "forget_thr: ", forget_thr
			print ""

	
	def Show(self):
		print ""
		print "ARTNet"
		print "n_in: ", self.n_in
		print "dissimilarity_func: ", self.dissimilarity_func
		print "rho: ", self.rho
		print "max_neurons: ", self.max_neurons
		print "forget_thr: ", self.forget_thr
		print ""
		
	def PerformanceFunction(self,input,cluster):
		dissimilarity = 9999
		
		if self.dissimilarity_func == "euc":
			dissimilarity = np.sum((input-cluster)**2)
		
		return dissimilarity
	
	def CreateNeuron(self, input):
		# check if is necessary
		isnecessary = False
		
		# if there is no neuron -> is necessary
		if self.neurons == None:
			isnecessary = True
			
		# if the event is not close of any previous neurons -> is necessary
		else:
			for ineuron in range(self.neurons.shape[0]):
				if self.PerformanceFunction(input,self.neurons[ineuron,:]) > self.rho:
					isnecessary = True
		
		if (isnecessary) and (self.neurons.shape[0] < self.max_neurons):
			self.neurons.append(input)
			self.neurons_up.append(0)
			
	def CloserNeuron(self, input):
		# Return the index of closer neuron and its dissimilarity
		closer_id = -9999
		dissi_value = 99999999
		
		for ineuron in range(self.neurons.shape[0]):
			if self.PerformanceFunction(input,self.neurons[ineuron,:]) < self.rho:
				if self.PerformanceFunction(input,self.neurons[ineuron,:]) < dissi_value:
					dissi_value = self.PerformanceFunction(input,self.neurons[ineuron,:])
					closer_id = ineuron
		return [closer_id, dissi_value]
	
	def DeleteNeuron(self,ineuron):
		# Remove the neuron with index ineuron
		
		#check if there is any neuron 
		if ineuron > self.neurons.shape[0]:
			print "ERROR in DeleteNeuron()"
			return -1
		np.delete(self.neurons,ineuron)
					
	def Train(self, input, trn_params=None):
		""" Train Function """
		
		""" 
    	Params:
    	
    	input: matrix
    	desc: input matrix
    	
    	trn_params: TrainParameters
    	desc: Train Parameters
    	
    	"""
    	
		if trn_params == None:
			self.trn_params = TrainParameters()
			
			ss = ShuffleSplit(input.shape[1], n_iter=1, test_size=self.trn_params.perc_tst)
			
			for train, test in ss:
				self.trn_params.itrn = train
				self.trn_params.itst = test

		else:
			self.trn_params = trn_params
			
		if self.neurons == None:
			self.CreateNeuron(input[:,self.trn_params.itrn[0]])
		
		for ievent in len(self.trn_params.itrn):
			[closer_id, dissi_value] = self.CloserNeuron(input[:,ievent])

			
			if closer_id == -9999:
				# no closer neuron
				self.CreateNeuron(input[:,self.trn_params.itrn[ievent]])
			else:
				# update neuron
				diff = self.neurons[closer_id,:]-input[:,self.trn_params.itrn[ievent]]
				self.neurons[closer_id,:] = (self.neurons[closer_id,:]+ self.trn_params.learning_rate*diff)
				self.neurons_up[closer_id] = 0
		
		# Forget Neurons
		for ineuron in range(self.neurons.shape[0]):
			if self.neurons_up[icluster] > self.forget_thr:
				self.DeleteNeuron(ineuron)
				
			