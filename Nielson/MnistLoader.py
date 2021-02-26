import gzip
import matplotlib as plt
import numpy as np
import random


def createBatches(x,y,sizeBatch):
    """Randomly shuffle the data"""
    data = list(zip(x,y))
    random.shuffle(data)
    
    ii = 0
    dataBatch  = []
    labelBatch = []
    # Create 
    for x,y in data:
        ii += 1
        if ii == 1:
            dataMatrix = np.reshape(x,(784,1))
            labelMatrix = y
        else:                
            dataMatrix = np.append(dataMatrix,np.reshape(x,(784,1)),axis=1)
            labelMatrix = np.append(labelMatrix,y,axis=1)
            if ii == sizeBatch:
               dataBatch.append(dataMatrix) 
               labelBatch.append(labelMatrix)
               ii = 0
    if ii != sizeBatch:
       dataBatch  = dataBatch[:-1]
       labelBatch = labelBatch[:-1]
    return dataBatch, labelBatch

def readBinaryFileToImage(filename, bytes2skip, numImage=[]):
    fzip  = gzip.open(filename, 'rb')
    file  = list(fzip.read())
    file  = file[bytes2skip:]
    image = []
    
    if numImage:
        numImage = numImage*784
    else:
        numImage = len(file)
    if len(file)    <   numImage:
        print('error: more image are requeted than the number of images in the file')
        return
    
    for ii in range(0,numImage,784):
        image.append( np.reshape(np.array(file[ii:ii+784]), (28,28)) )
    return image

def readBinaryFileToLabel(filename, bytes2skip,numImage=[]):
    fzip = gzip.open(filename, 'rb')
    file = list(fzip.read())
    
    if numImage:
        if len(file)    <   (numImage - bytes2skip):
            print('error: requested number of label is larger than nuber of labels in the file')
            return
        else:
            file = file[bytes2skip:numImage+bytes2skip]
    else:
        file = file[bytes2skip:]
    y = []
    for ii in file:
        temp = np.zeros((10,1))
        temp[ii] = 1
        y.append(temp)
    return y


def createDataMatrix(images,labels):
    data = zip(images,labels)
    ii = 0
    for x,y in data:
        if ii == 0:
            imageMatrix = np.reshape(x,(784,1))
            labelMatrix = y
            ii =2
        else:                
            imageMatrix = np.append(imageMatrix,np.reshape(x,(784,1)),axis=1)
            labelMatrix = np.append(labelMatrix,y,axis=1)
    return  imageMatrix, labelMatrix

def createMinBtach(images,labels):
    data = list(zip(images,labels))
    random.shuffle(data)
    images,labels  = zip(*data)
    images,labels  = createDataMatrix(images,labels )
    return images,labels 

def vector2number(vec):
    ii = 0
    for x in vec: 
        if x == 1:
            break
        else:
            ii += 1
    return ii

def readBinaryFile2testDataMatrix(filename, bytes2skip, numImage=[],loadingTuple=[]):
    fzip  = gzip.open(filename, 'rb')
    file  = list(fzip.read())
    if numImage:
        numImage = numImage*784
    else:
        numImage = len(file)-bytes2skip
    if len(loadingTuple) > 1:
        print(loadingTuple[0])
    file  = file[bytes2skip:numImage+bytes2skip]
    matrix  = np.reshape(np.array(file), (int(len(file)/784),784))
    if len(loadingTuple) > 1:
        print(loadingTuple[1])
    return matrix.transpose()

def readBinaryFile2Label(filename, bytes2skip,numImage=[],loadingTuple=[]):
    fzip = gzip.open(filename, 'rb')
    file = list(fzip.read())
    if numImage:
        numImage = numImage
    else:
        numImage = len(file)-bytes2skip
    if len(loadingTuple) > 1:
        print(loadingTuple[0])
    file = file[bytes2skip:numImage+bytes2skip]
    labelMatrix = np.zeros((10,len(file)))
    labelMatrix[file,np.arange(0,len(file),1)] = 1
    if len(loadingTuple) > 1:
        print(loadingTuple[1])
    return labelMatrix


# Network with nearly 90 result:
#path = 'D:/Master/'
#numImages = 60000
#Batchsize = 30
#eta       = 0.3
#epochs    = 30
#NetwotkNeurons = [784,40,10]    

path = 'D:/Master/'
numImages = 60000
Batchsize = 30
eta       = 0.3
epochs    = 30
NetwotkNeurons = [784,40,10]

# loading data and decoding
testImage = readBinaryFile2testDataMatrix(path+'t10k-images-idx3-ubyte.gz',16,loadingTuple=['Loading test data images...','Loading done']) 
TestLabel = readBinaryFile2Label(path+'t10k-labels-idx1-ubyte.gz',8,loadingTuple=['Loading test data labels...','Loading done']) 
trainImages = readBinaryFile2testDataMatrix(path+'train-images-idx3-ubyte.gz',16,numImage=numImages,loadingTuple=['Loading test data images...','Loading done'])
trainLabels = readBinaryFile2Label(path+'train-labels-idx1-ubyte.gz',8,numImage=numImages,loadingTuple=['Loading test data labels...','Loading done'])

# cretae test image randomly selected from the test image data set to verify decoding
imageNum = round(random.uniform(0, len(TestLabel)/10))
a = np.reshape(testImage[:,imageNum],(28,28))
plt.pyplot.imshow(a,cmap='gray', vmin=0, vmax=255)
print('Display test image and label to verify decoding:')
print('Number in image is: '+str(vector2number(TestLabel[:,imageNum])))
print('Creating Test Data Done')
print(testImage.shape)
print(TestLabel.shape)



# Create neural network and train the network
from NeuralNetworkFast import NeuralNetwork
NetworkTest = NeuralNetwork(NetwotkNeurons)
NetworkTest.sgd(trainImages,trainLabels,Batchsize,epochs,eta,testImage,TestLabel)















