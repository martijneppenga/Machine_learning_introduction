# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:55:11 2020

@author: martijn
This class is designed to load the data from the mnist handwritten images 
library. 
To use this class the t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz,
train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz data set are needed.
These data sets can be found at http://yann.lecun.com/exdb/mnist/.

The class has methods to decode the data in these data files and retun the data
as 2 dimensional numpy arrays.

The MNIST data set contains images of handwritten numbers and labels for each 
image containg the number in the image. The image are gray scale and have a 
format of 28x28
"""
import gzip
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataLoader:
    
    def __init__(self):
        self.TestImage  = 't10k-images-idx3-ubyte.gz'
        self.TestLabel  = 't10k-labels-idx1-ubyte.gz'
        self.TrainImage = 'train-images-idx3-ubyte.gz'
        self.TrainLabel = 'train-labels-idx1-ubyte.gz'
        return
    
    def loadAll(self,path=''):
        '''Load all images and lables of the MNIST data set
        returns four numpy arrays:
            Input path: the path to the folder where the images and labels are
            stored
            testImage: array of 784x10000 with 10000 images
            testLabel: array of 10x10000 with 10000 labels for the test set
            trainImages: array of 784x60000 with 600000 images
            trainlables: array of 10*60000 with 600000 labels for the train set
            The images are gray sacle with a value between 0 to 255 (int32)
            The labels are vectors with zeros and a 1 at the index corresponding
            to the number in the image (float64)'''
        testImage   = self.readBinaryFile2DataMatrix('Test', path, 16) 
        testLabel   = self.readBinaryFile2LabelMatrix('Test', path, 8) 
        trainImages = self.readBinaryFile2DataMatrix('Train', path, 16)
        trainLabels = self.readBinaryFile2LabelMatrix('Train', path, 8)
        return testImage, testLabel, trainImages, trainLabels

    
    def labelVector2number(self,vec):
        """Function to convert a output vector of the neuron network the the 
        recoginsed number"""
        ii = 0
        for x in vec: 
            if x == 1:
                break
            else:
                ii += 1
        return ii
    

    def readBinaryFile2DataMatrix(self,dataset,path='', bytes2skip=16, numImage=None,):
        '''This function retuns a matrix with the images from the mnist data set
        The colums of the matrix contains the data of one image. You can plot 
        the images using the showImage method. Or one can make a plot of the image
        by reshaping the colum to a 28x28 matrix
        The dataset tells the function which data set will be loaded:
            options: "Test"  (file with 10000 images), 
                     "Train" (file with 60000 images)
        Path specifies the search path to the image folder
        The bytes2skip input skip the specified number of bytes from the start 
        of the file. The default is 16 as the first image start at byte 17
        The numImage input tells the function how many images must be loaded
        All image will be loaded if this variable is left empty
        To get the corresponding label of the images one should use the method
        readBinaryFile2LabelMatrix with the same input as used for this function
        However, keep in mind that there is an offset -8 in the bytes2skip variable
        '''
           
        if len(path) > 0:
            if path[-1] != '/':
                path = path+'/'
                
        if dataset == 'Train':
            filename = path+self.TrainImage
            loadmessage = ['Loading training data images...','Loading done']
        elif dataset == 'Test':
            filename = path+self.TestImage
            loadmessage = ['Loading testing data images...','Loading done']
        else:
            print('Could not find requeste data set')
            return
        
        fzip  = gzip.open(filename, 'rb')
        file  = list(fzip.read())
        if numImage:
            numImage = numImage*784
        else:
            numImage = len(file)-bytes2skip

        print(loadmessage[0])
            
        file  = file[bytes2skip:numImage+bytes2skip]
        matrix  = np.reshape(np.array(file), (int(len(file)/784),784))
        
        print(loadmessage[1])
        return matrix.transpose()
    
    def readBinaryFile2LabelMatrix(self,dataset,path='', bytes2skip=8,numImage=None):
        '''This function retuns a matrix with the labels for the mnist data set
        The colums of the matrix contains the data of one image. This colums will
        have a length of 10 and consist out of zeros. Only the index of 
        corresponding to the label number will contain a one: example if the label
        of an image is 5 then a one will be found on index number 5 
        The dataset tells the function which data set will be loaded:
            options: "Test"  (file with 10000 labels), 
                     "Train" (file with 60000 labels)
        Path specifies the search path to the image folder
        The bytes2skip input skip the specified number of bytes from the start 
        of the file. The default is 16 as the first image start at byte 17
        The numImage input tells the function how many images must be loaded
        All image will be loaded if this variable is left empty
        To get the corresponding images by the labels one should use the method
        readBinaryFile2DataMatrix with the same input as used for this function
        However, keep in mind that there is an offset +8 in the bytes2skip variable'''
        
        if len(path) > 0:
            if path[-1] != '/':
                path = path+'/'
                
        if dataset == 'Train':
            filename = path+self.TrainLabel
            loadmessage = ['Loading training data labels...','Loading done']
        elif dataset == 'Test':
            filename = path+self.TestLabel
            loadmessage = ['Loading testing data labels...','Loading done']
        else:
            print('Could not find requeste data set')
            return
        fzip = gzip.open(filename, 'rb')
        file = list(fzip.read())
        if numImage:
            numImage = numImage
        else:
            numImage = len(file)-bytes2skip
        
        print(loadmessage[0])
            
        file = file[bytes2skip:numImage+bytes2skip]
        labelMatrix = np.zeros((10,len(file)))
        labelMatrix[file,np.arange(0,len(file),1)] = 1
        
        print(loadmessage[1])
        return labelMatrix
    

    def showImage(self,Image,Label=None,ImageNumber=None):
        "Display image. Input should have a length of 784 i.e. one image"
        Image = np.reshape(Image,(28,28))
        plt.imshow(Image,cmap='gray', vmin=0, vmax=255)
        if len(Label):
            if ImageNumber:
                plt.title('Number in image is: '+str(self.labelVector2number(Label))+
                          '\n Image number: '+str(ImageNumber))
            else:
                plt.title('Number in image is: '+str(self.labelVector2number(Label)))
        return