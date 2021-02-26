# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:51:18 2020
http://neuralnetworksanddeeplearning.com/chap1.html
@author: Martijn
"""

import numpy as np
import random
from datetime import datetime

class NeuralNetwork:
        
    def __init__(self, Neurons):
        """Input is list containing of lenght >= 2. The first element represents
        the number of input neurons, second element represents the number of 
        neurons in the second layer. This goes on to the last element within 
        the input list"""
        self.NumLayers = len(Neurons)
        self.Neurons   = Neurons
        self.Biases    = [np.random.randn(y, 1) for y in Neurons[1:]]
        self.Weights   = [np.random.randn(y, x)*1/np.sqrt(x) for x, y in zip(Neurons[:-1], Neurons[1:])]
        self.FileName  = self.CreateFileName()
    
    def saveNetwork(self,Filename):
        np.save(Filename+'NumLayers',self.NumLayers)
        np.save(Filename+'Neurons',self.Neurons)
        np.save(Filename+'Biases',self.Biases)
        np.save(Filename+'Weights',self.Weights)
        return
    
    def loadNetwork(self,Filename):
        self.NumLayers = np.load(Filename+'NumLayers.npy')
        self.Neurons   = np.save(Filename+'Neurons.npy')
        self.Biases    = np.save(Filename+'Biases.npy')
        self.Weights   = np.save(Filename+'Weights.npy')
        return
        
    def getWeightsBias(self):
        return self.Weights, self.Biases
    
    def saveWeightsBias(self, FileName):
        w,b = self.getWeightsBias()
        np.save(FileName+'Weights',w)
        np.save(FileName+'Biases',b)
        return
    
    def loadWeightsBias(self,Filename):
        self.Weights = np.load(Filename+'Weights.npy')
        self.Biases  = np.load(Filename+'Biases.npy')
        return

    def sigmoid(self,x):
        # Sigmoid function output between [-1, 1]
        return 1/(1+np.exp(-x))
    
    def derivativeSigmoid(self,x):
        # Derivative of the sigmoid function
        temp = self.sigmoid(x)
        return temp*(1-temp)
    
    def cost_derivative(self, aL, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (aL-y)
    
    def feedforward(self, a):
        """Return the output of the network if "a" is input.
        the input a can be a vector or a matrix."""
        if len(a.shape) == 1:
            a = np.reshape(a,(self.Neurons[0],1))
        for b, w in zip(self.Biases, self.Weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
    
    def backprop(self,x,y):
        """x is a matrix with the collums representing the input for a neural 
        network and the rows representing the inputs for multiple runs.
        The frunction returns a tuple with in each element the sum of the gradiant
        for the weights and biases for the multiple runs"""
        if len(x.shape) == 1:
            x = np.reshape(x,(self.Neurons[0],1))
            y = np.reshape(y,(self.Neurons[-1],1))
        activation        = x
        NeuronsActivation = [x]
        z                 = []
        grad_b = [np.zeros(b.shape) for b in self.Biases]
        grad_w = [np.zeros(w.shape) for w in self.Weights]
        for w,b in zip(self.Weights,self.Biases):
            """Calculate z vector for each layer and the acivation of each neuron
            in that layer"""
            x = np.dot(w,activation) + b
            z.append(x)
            activation = self.sigmoid(x)
            NeuronsActivation.append(activation)

        delta = self.cost_derivative(NeuronsActivation[-1], y) * self.derivativeSigmoid(z[-1])
        grad_b[-1] = np.reshape(np.sum(delta,axis=1),(self.Neurons[-1],1))
        grad_w[-1] = np.dot(delta, NeuronsActivation[-2].transpose())
        
        for ii in range(2,self.NumLayers):
            delta       = np.dot(self.Weights[-ii+1].transpose(),delta) * \
                            self.derivativeSigmoid(z[-ii])
            grad_b[-ii] = np.reshape(np.sum(delta,axis=1),(self.Neurons[-ii],1))
            grad_w[-ii] = np.dot(delta,NeuronsActivation[-ii-1].transpose())
        return (grad_w, grad_b)
        
    def CreateFileName(self):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        dt_string = 'NeuralNetworkResults_'+dt_string+'.txt'
        return dt_string
        

    def saveResult(self,text='',numimagesTrain=0,numimagesTest=0,eta=0,batchSize=0,epochs=0):
        """This function creates a text file with the relevant parameters of the
        network and the training result of the network after epoch. If no 
        validatio data is supplied to the network than instead of the network
        result a list of the number op epoch will be save
        For the MNIST data set, it turned out that a normalized input improved
        the results significantly"""
        file = open('Results/'+self.FileName,'a+')
        if len(text) == 0:
            file.write('Network '+str(self.Neurons)+'\n')
            file.write('Number of training images: '+str(numimagesTrain)+'\n')
            file.write('Number of test images: '+str(numimagesTest)+'\n')
            file.write('Learning rate: eta = '+str(eta)+'\n')
            file.write('Batch size: '+str(batchSize)+'\n')
            file.write('Number of epochs: '+str(epochs)+'\n')
            file.write('\n')
            file.write('Results:\n')
        else:
            file.write(text+'\n')
            
        file.close()
        return
    
   
    
    def updateWeights(self,gradW,gradB,eta,sizeBatch):
        self.Weights = [w - (eta/sizeBatch)*nw for w, nw in zip(self.Weights, gradW)]
        self.Biases  = [b - (eta/sizeBatch)*nb for b, nb in zip(self.Biases, gradB)]
        return
    
    def sgd(self,x,y,sizeBatch,epoch,eta,testData=[],testLabel=[],saveResults = False):
        """Gradiant descent training function. The function trains a neural network
        using the x and y data as input, where the x data is a matrix with the input 
        for the network in its colums and y is a matrix with the requested result 
        of the output in its colums (i.e a label matrix).
        The function makes batches from the input data when training.
        and will loop epoch times over the entire dataset
        You can supply the algorithm with testdata and testlabel. These inputs
        are used to validate the network after each epoch. The data matrices use
        the same structure as the x and y matrices
        saveResults will save the performance of the network in a text file"""
        dim = x.shape
        indexVector = np.arange(0,dim[1],1,dtype='int')
        if saveResults:
            nTrain = dim[1]
            try:
                dimTest = testData.shape
                nTest = dimTest[1]
            except:
                nTest = 0 
            self.saveResult('', nTrain, nTest, eta, sizeBatch, epoch)
        if dim[1] % sizeBatch != 0:
            n = dim[1] - (dim[1] % sizeBatch)
        else:
            n = dim[1]
        for ii in range(epoch):
            #Create a index vector and shuffle it to take random elements from 
            #the input matrix for the batches 
            xdata = np.zeros((self.Neurons[0],sizeBatch))
            ydata = np.zeros((self.Neurons[-1],sizeBatch))
            random.shuffle(indexVector)
            for k in range(0,n,sizeBatch):
                # train network for each batch of data using gradient descent
                xdata[:,:] = x[:,indexVector[k:k+sizeBatch]]
                ydata[:,:] = y[:,indexVector[k:k+sizeBatch]]
                gradW, gradB = self.backprop(xdata,ydata)
                self.updateWeights(gradW, gradB, eta, sizeBatch)
            if len(testData)>0:
                Result = self.evaluate(testData,testLabel)
                print('Epoch number %3d: Test result: %1.4f' % (ii,Result ))
                if saveResults:
                    self.saveResult('Epoch number %3d: Test result: %1.4f' % (ii, Result))
            else:
                print('Epoch number %3d: complete' % ii)
                if saveResults:
                    self.saveResult('Epoch number %3d: complete' % ii)
            
        return
    
    def evaluate(self,xdata,ydata):
        """This function evaluates the perfomrance of a neuron network.
        The "xdata" input is a numpy data matrix with the data for the first neuron layer.
        The input for the first neuron is read as xdata[:,n], where n is the 
        nth test data set.
        The "ydata" matrix contains the desired result from the neuron network"""
        label       = self.feedforward(xdata)
        numberImage = np.argmax(label, axis=0)
        numberLabel = np.argmax(ydata, axis=0)
        value = np.sum(np.equal(numberImage,numberLabel))/len(numberImage)       
        return value
    
    def  NetworkOutput2Number(self,y):
        """This function returns the element with the higest value for column
        of the inpt matrix"""
        if len(y) == self.Neurons[-1]:
            y = np.reshape(y,(self.Neurons[-1],1))
        return np.argmax(y, axis=0)
    
        
        
        
        
        
        
        
        
        
        