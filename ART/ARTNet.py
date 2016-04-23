import os
import sys
import timeit

import numpy
import numpy as np

import random

from sklearn.cross_validation import ShuffleSplit

class TrainParameters(object):
	def __init__(self, learning_rate=0.01, perc_trn=0.5, perc_tst=0.5, show=False):
		""" 
			Train Parameters Class 
			
			learning_rate: Learning Rate
			perc_trn: Percentage of Train Events
			perc_tst: Percentage of Test Events
		"""
		self.learning_rate = learning_rate
		
		self.perc_trn = perc_trn
		self.perc_tst = perc_tst
		
		self.itrn = None
		self.itst = None
		
		if show:
			self.Show()
		
	def Show(self):
		print "Train Parameters" 
		print "Learning Rate: ", self.learning_rate
		print "Percentage Train Events: ", self.perc_trn
		print "Percentage Test Events: ", self.perc_tst
		print "Train Indeces: ", self.itrn
		print "Test Indeces: ", self.itst
		print ""
				
class ART(object):
	""" ART Net """
	
	def __init__(self, n_in, n_out, dissimilarity_func='euc', rho=0.1, max_neurons=10, forget_thr=10, trn_params=None, show=False):
	
		""" 
    	Params:
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
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
		self.n_out = n_out
		self.dissimilarity_func = dissimilarity_func
		self.rho = rho
		self.max_neurons = max_neurons
		self.forget_thr = forget_thr
		
		self.activate_neurons = -9999*np.ones([1,self.n_in])
		self.last_updates = -1
		self.qtd_activate_neurons = 0
		
		self.forgot_neurons = None
		self.qtd_forgot_neurons = 0
		
		self.class_id = 0
		
		if trn_params == None:
			self.trn_params = TrainParameters()
		else:
			self.trn_params = trn_params
		
		if show:
			self.Show()
	
	def Show(self):
		print ""
		print "ARTNet"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		
		if self.dissimilarity_func == "euc":
			print "Dissimilarity Function: Euclidian"
		if self.dissimilarity_func == "euc_norm":
			print "Dissimilarity Function: Normalized Euclidian"
		
		print "rho: ", self.rho
		print "max_neurons: ", self.max_neurons
		print "forget_thr: ", self.forget_thr
		print ""
		self.trn_params.Show()
		print ""
		print "Active Neurons: ", self.activate_neurons
		print "Quantity of Active Neurons: ", self.qtd_activate_neurons
		print "Last Update: ", self.last_updates
		print ""
		print "Forgot Neurons: ",self.forgot_neurons
		print "Quantity of Forgot Neurons: ", self.qtd_forgot_neurons		
		print ""
		print "Classification Index: ", self.class_id
		print ""
		
	def PerformanceFunction(self,vector1,vector2):
		#print "Performance Function"
		"""
			This Function will measure the dissimilarity between two vectors
		"""
		
		#print "Shape[vector1]: ", vector1.shape
		#print "Shape[vector2]: ", vector2.shape
		
		if not (vector1.shape == vector2.shape):
			print "Vectors with different sizes"
			return False
		
		if self.dissimilarity_func == "euc":
			dissimilarity = np.sum((vector1-vector2)**2)
		if self.dissimilarity_func == "euc_norm":
			dissimilarity = np.sum((vector1-vector2)**2)/vector1.shape[0]
		
		#print "Dissimilarity value: ", dissimilarity
		return dissimilarity
	
	def CreateNeuron(self, input):
		#print "Function Create Neuron"
		
		"""
			This function will create a new neuron if it is possible
		"""			
		if not (len(input) == self.n_in):
			print "input has a different dimension of self.n_in"
			return False

		# Check if is possible to create this new neuron
		
		# if there is no activate neuron -> Create a new neuron
		if self.qtd_activate_neurons == 0:
			#print "Creating the First Neuron"
			for id in range(self.n_in):
				self.activate_neurons[:,id] = input[id]
			self.qtd_activate_neurons = 1
			self.last_updates = 0
			return True

		# if there any neuron with an smaller dissimilarity -> Do not create
		isnecessary = True
		for ineuron in range(self.qtd_activate_neurons):
			neuron = self.activate_neurons[ineuron,:]
			dissimilarity = self.PerformanceFunction(neuron,input)
			 
			if dissimilarity < self.rho:
				isnecessary = False
				break

		# Create the new neuron
		if isnecessary:
			
			buffer_input = np.ones([1,self.n_in])
			for id in range(self.n_in):
				buffer_input[:,id] = input[id]
			
			self.activate_neurons = np.append(self.activate_neurons,buffer_input,axis=0)
			self.last_updates = np.append(self.last_updates,0)
			self.qtd_activate_neurons = self.qtd_activate_neurons+1
			
		return isnecessary
		
	def CloserNeuron(self, input):
		print "Function CloserNeuron"
		"""
			This function will return the index of closer neuron and dissimilarity value
		"""
		closer_neuron = -1
		smaller_dissim = 9999999
		
		# check in all activate neurons
		for ineuron in range(self.qtd_activate_neurons):
			neuron = self.activate_neurons[ineuron,:]
			dissimilarity = self.PerformanceFunction(neuron,input)
			if dissimilarity < smaller_dissim:
				smaller_dissim = dissimilarity
				closer_neuron = ineuron
				
		# maybe check in inactive?
							
		return [closer_neuron, smaller_dissim]
		
	def CloserForgotNeuron(self, input):
		print "Function CloserForgotNeuron"
		"""
			This function will return the index of closer forgot neuron and dissimilarity value
		"""
		closer_neuron = -1
		smaller_dissim = 9999999
		
		# check in all activate neurons
		for ineuron in range(self.qtd_forgot_neurons):
			neuron = self.forgot_neurons[ineuron,:]
			dissimilarity = self.PerformanceFunction(neuron,input)
			if dissimilarity < smaller_dissim:
				smaller_dissim = dissimilarity
				closer_neuron = ineuron
				
		# maybe check in inactive?
							
		return [closer_neuron, smaller_dissim]
	
	def ForgetNeuron(self,ineuron):
		#print "Function ForgetNeuron"
		"""
			This function will forget a inactive neuron
		"""
		
		# Check if can forget this neuron
		if ineuron > self.qtd_activate_neurons:
			print "Can not forget this neuron: ",ineuron
			return False
		
		# if there is no inactivate neuron -> Create a new inactive neuron
		if self.qtd_forgot_neurons == 0:
			#print "Creating the First Inactivate Neuron"
			
			buffer_neuron = np.ones([1,self.n_in])
			for id in range(self.n_in):
				buffer_neuron[:,id] = self.activate_neurons[ineuron,id]
			
			self.forgot_neurons = buffer_neuron
			self.qtd_forgot_neurons = 1
			self.activate_neurons = np.delete(self.activate_neurons,ineuron,axis=0)
			self.qtd_activate_neurons = self.qtd_activate_neurons-1
			self.last_updates = np.delete(self.last_updates,ineuron,axis=0)
			return True
		
		else:
			buffer_neuron = np.ones([1,self.n_in])
			for id in range(self.n_in):
				buffer_neuron[:,id] = self.activate_neurons[ineuron,id]
				
			self.forgot_neurons = np.append(self.forgot_neurons,buffer_neuron,axis=0)
			self.qtd_forgot_neurons = self.qtd_forgot_neurons+1
			self.activate_neurons = np.delete(self.activate_neurons,ineuron,axis=0)
			self.qtd_activate_neurons = self.qtd_activate_neurons-1
			self.last_updates = np.delete(self.last_updates,ineuron)
			return True
		
		return False
		
	def ReactivateNeuron(self,ineuron):
		print "Function ReactivateNeuron"
		"""
			This function will reactivate a inactive neuron
		"""
		# Check if ineuron belongs to inactive neurons
		
		if ineuron > self.qtd_forgot_neurons:
			print "Can not reativate this neuron: ", ineuron
			return False
			
		# if there is no activate neuron -> Create a new neuron
		if self.qtd_activate_neurons == 0:
			#print "Creating the First Neuron"
			
			buffer_neuron = np.ones([1,self.n_in])
			for id in range(self.n_in):
				buffer_neuron[:,id] = self.forgot_neurons[ineuron,id]
			
			self.activate_neurons = buffer_neuron
			self.qtd_activate_neurons = 1
			self.last_updates = 0
			self.qtd_forget_neurons = self.qtd_forgot_neurons-1
			return True
 
		else:
			# print "Reactivating a Neuron"
			buffer_neuron = np.ones([1,self.n_in])
			for id in range(self.n_in):
				buffer_neuron[:,id] = self.forgot_neurons[ineuron,id]
			
			self.activate_neurons = np.append(self.activate_neurons,buffer_neuron,axis=0)
			self.qtd_activate_neurons = self.qtd_activate_neurons+1
			self.last_updates = np.append(self.last_updates,0)
			self.forgot_neurons = np.delete(self.forgot_neurons,ineuron,axis=0)
			self.qtd_forgot_neurons = self.qtd_forgot_neurons-1
			return True
		return False
	
	def UpdateNeuron(self,ineuron,input):
		print "Function UpdateNeuron"
		"""
			This function will update a neuron value
		"""
		# check if this is a valid neuron
		if ineuron > self.qtd_activate_neurons:
			print "Not valid neuron"
			return False

		# check dimensionality
		if not (len(input) == self.n_in):
			print "input has a different dimension of self.n_in"
			return False

		buffer_neuron = np.ones([1,self.n_in])
		for id in range(self.n_in):
			buffer_neuron[:,id] = self.activate_neurons[ineuron,id]
		buffer_input = np.ones([1,self.n_in])
		for id in range(self.n_in):
			buffer_input[:,id] = input[id]
		
		update = buffer_neuron - buffer_input
		#print "Neuron: ", buffer_neuron
		#print "Input: ", buffer_input
		#print "Update: ", update
		
		buffer_neuron = buffer_neuron+self.trn_params.learning_rate*update
		
		#print "New Neuron: ", buffer_neuron
		
		self.activate_neurons[ineuron,:] = buffer_neuron
		return True
	
	def PaintNeurons(self, inputs, targets):
		print "PaintNeurons"
		"""
			This function will paint the neurons 
			(check the percentage of each class in each neuron)
		"""
		# como fazer isso sem ter que voltar atras e repensar tudo de novo????
		
	def Train(self, input, target, trn_params=None):
		print  "Train Function"
		
		"""
			This function will train the model
		"""
		
		return False