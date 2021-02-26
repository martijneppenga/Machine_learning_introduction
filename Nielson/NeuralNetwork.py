# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:51:18 2020
http://neuralnetworksanddeeplearning.com/chap1.html
@author: Martijn
"""

import numpy as np
import random

class NeuralNetwork:
        
    def __init__(self, Neurons):
        """Input is list containing of lenght >= 2. The first element represents
        the number of input neurons, second element represents the number of 
        neurons in the second layer. This goes on to the last element within 
        the input list"""
        self.NumLayers = len(Neurons)
        self.Neurons   = Neurons
        self.Biases    = [np.random.randn(y, 1) for y in Neurons[1:]]
        self.Weights   = [np.random.randn(y, x) for x, y in zip(Neurons[:-1], Neurons[1:])]
    
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
        
        NeuronsActivation = [x]
        z                 = []
        grad_b = [np.zeros(b.shape) for b in self.Biases]
        grad_w = [np.zeros(w.shape) for w in self.Weights]
        ii     = 0
        for w,b in zip(self.Weights,self.Biases):
            """Calculate z vector for each layer and the acivation of each neuron
            in that layer"""
            x = np.dot(w,NeuronsActivation[ii]) + b
            z.append(x)
            NeuronsActivation.append(self.sigmoid(x))
            ii += 1
        delta = self.cost_derivative(NeuronsActivation[-1], y) * self.derivativeSigmoid(z[-1])
        grad_b[-1] = np.reshape(np.sum(delta,axis=1),(self.Neurons[-1],1))
        grad_w[-1] = np.dot(delta, NeuronsActivation[-2].transpose())
        
        for ii in range(2,self.NumLayers):
            delta       = np.dot(self.Weights[-ii+1].transpose(),delta) * \
                            self.derivativeSigmoid(z[-ii])
            grad_b[-ii] = np.reshape(np.sum(delta,axis=1),(self.Neurons[-ii],1))
            grad_w[-ii] = np.dot(delta,NeuronsActivation[-ii-1].transpose())
        return (grad_w, grad_b)
        
    
    def miniBatches(self,data,batchSize):
        """return data separated in batches random picked from the input data"""
        n = len(data)
        random.shuffle(data)
        return [data[k:k+batchSize] for k in range(0, n, batchSize)]
    
    
    def createDataMatrix(self,images,labels):
        print('Create data matrices')
        ii = 0
        for x,y in zip(images,labels):
            if ii == 0:
                imageMatrix = np.reshape(x,(self.Neurons[0],1))
                labelMatrix = y
                ii =2
            else:                
                imageMatrix = np.append(imageMatrix,np.reshape(x,(self.Neurons[0],1)),axis=1)
                labelMatrix = np.append(labelMatrix,y,axis=1)
        print('Create data matrices done')
        return  imageMatrix, labelMatrix
        
    def createBatches(self,x,y,sizeBatch):
        """Randomly shuffle the data"""
        print('Creating batches')
        data = list(zip(x,y))
        random.shuffle(data)
        
        ii = 0
        dataBatch  = []
        labelBatch = []
        # Create tuple with data and label matrix with the number of colums equal to
        # the size of sizeBatch 
        for x,y in data:
            ii += 1
            if ii == 1:
                dataMatrix = np.reshape(x,(self.Neurons[0],1))
                labelMatrix = y
            else:                
                dataMatrix  = np.append(dataMatrix,np.reshape(x,(self.Neurons[0],1)),axis=1)
                labelMatrix = np.append(labelMatrix,np.reshape(y,(self.Neurons[-1],1)),axis=1)
                if ii == sizeBatch:
                   dataBatch.append(dataMatrix) 
                   labelBatch.append(labelMatrix)
                   ii = 0
        if ii != sizeBatch:
           dataBatch  = dataBatch[:-1]
           labelBatch = labelBatch[:-1]
        print('Creating batches done')
        return dataBatch, labelBatch
    
    def updateWeights(self,gradW,gradB,eta,sizeBatch):
        self.Weights = [w - (eta/sizeBatch)*nw for w, nw in zip(self.Weights, gradW)]
        self.Biases  = [b - (eta/sizeBatch)*nb for b, nb in zip(self.Biases, gradB)]
        return
    
    def sgd(self,x,y,sizeBatch,epoch,eta,testData=[],testLabel=[]):
        for ii in range(epoch):
            xdata, ydata = self.createBatches(x,y,sizeBatch)
            for data,label in zip(xdata, ydata):
                gradW, gradB = self.backprop(data,label)
                self.updateWeights(gradW, gradB, eta, sizeBatch)
            if len(testData)>0:
                print('Epoch number %5d: test result: %1.4f' % (ii, self.evaluate(testData,testLabel)))
            else:
                print('Epoch number  %5d: complete' % ii)
            
        return
    
    def evaluate(self,xdata,ydata):

        label       = self.feedforward(xdata)
        numberImage = np.argmax(label, axis=0)
        numberLabel = np.argmax(ydata, axis=0)
        value = np.sum(np.equal(numberImage,numberLabel))/len(numberImage)       
        return value
    
        
        
        
        
        
        
        
        
        
        