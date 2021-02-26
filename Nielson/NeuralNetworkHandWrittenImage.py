# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:21:40 2020
http://neuralnetworksanddeeplearning.com/chap1.html
@author: martijn
"""
import matplotlib.pyplot as plt
import numpy as np
import random

# load classes
from MNISTDataLoader import MNISTDataLoader
from NeuralNetworkFast import NeuralNetwork

path = 'D:/Master/'
numImagesTrain = 60000   # max 60000
numImagesTest  = 10000   # max 10000
batchSize      = 30      # number of images used before updating the weights of the network
eta            = 1       # learning rate
epochs         = 30      # number of times to loop trough the test images
NetworkNeurons = [784,30,20,10] # begin must be 784, end must be 10 everthing in between can be random
SaveResults2textfile = False # save network parameters and results in .txt file

# Create an image with some of the wornly identified numbers by the neural network
# The imagec contains the first numrow*numcol wronly identified images from the 
# testdata set
CreateWrongIdentifiedPicturs = True # True when one the image
numrow = 6 # number of rows within the subplot of wronly identified images
numcol = 4 # number of colums within the subplot of wronly identified images

# Loading and decoding data
dataloader  = MNISTDataLoader()
testImage   = dataloader.readBinaryFile2DataMatrix('Test',path,16,numImagesTest) 
testLabel   = dataloader.readBinaryFile2LabelMatrix('Test',path,8,numImage = numImagesTest) 
trainImages = dataloader.readBinaryFile2DataMatrix('Train',path,16,numImage = numImagesTrain)
trainLabels = dataloader.readBinaryFile2LabelMatrix('Train',path,8,numImage = numImagesTrain)

# Create neural network and train the network using stochastic gradient descent method
NetworkTest = NeuralNetwork(NetworkNeurons)
NetworkTest.sgd(trainImages/255,trainLabels,batchSize,epochs,eta,testImage/255,testLabel,SaveResults2textfile)

# Create image of wronly identified numbers
if CreateWrongIdentifiedPicturs:
    fig, axs = plt.subplots(numcol, numrow)
    col = 0
    row = 0
    a   = np.zeros((784,1))
    b   = np.zeros((10,1))
    
    for ii in range(0,numImagesTest):
        # This could be vectorized, but the code does not yet suffer from a slowdown
        a[:,:] = np.reshape(testImage[:,ii],(784,1))
        b[:,:] = np.reshape(testLabel[:,ii],(10,1))
        r      = NetworkTest.evaluate(a,b) 
        # returns a float, however as we only test one image we are certain the 
        # result is either a zero or a one (no floating point errors)
        if r == 0:
            # label and recognised number are diffrent, create image of this instance
            numberNeuralNetwork = NetworkTest.NetworkOutput2Number(NetworkTest.feedforward(a))[0]
            axs[col, row].imshow(np.reshape(a,(28,28)),cmap='gray', vmin=0, vmax=255)
            axs[col, row].set_title('Label: '+str(dataloader.labelVector2number(b))+
                      ' Network: '+str(numberNeuralNetwork))
            axs[col, row].axis('off')
            # update the colum and row indices of the subplot
            row += 1
            if (row % numrow) == 0:
                col += 1
                row = 0
                if (col % numcol) == 0:
                    # subplot cannot hold anymore images so break the loop
                    break

        
def showRandomlySelectedImage(testImage,testLabel):
    #shows a random image from the data set
    imageNum = testLabel.shape
    imageNum = round(random.uniform(0, imageNum[1]))
    dataloader.showImage(testImage[:,imageNum], testLabel[:,imageNum], imageNum)
    return
