# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:17:19 2020

@author: meppenga
This script is intended as an intorduction to machine learning.
The MNIST handwritten numbers database is use to train a machine learning model
for the classification of handwritten numbers.
The database consist of 60.000 training images and 10.000 validation images
displaying handwritten numbers between 0-9. The image size is 28x28 pixels.

Two different machine learning models are shown in this script:
    (FNN), a fully connected neural network
    (CNN), a convolutional neural network
The FNN model is a random model, while the CNN is a model from the Keras website.

This script encourages the user to play with it.  Note that some user input is
request when running this script (i.e. pop ups in commend line)

Few suggestions:
    Change the number of neurons of the FNN model (function FFNModel)
    Add extra layers to the FNN model
    Use a different activation function for the FNN model (user input)
    Change the number of epochs
    Chnage the learning rate
    Change the batch size
    Remove the data normalization (i.e. divide by 1 instead of 255) (try this one with FNN and activation sigmoid)
    Add regularization to FNN model (for instance a dropout layer)
    Chnage the learning rate (try something like 0.0005, 0.05 and 0.001)
    use a different optimizer (example SGD instead of Adam) FOR SGD USE A 
        DIFFERENT LEARNING RATE THAN FOR ADAM (something like learning_rate = 0,1)
    

Training optimzer: Adam
"""
import os
# Set tensorflow logging verbosity
# Ignore all info and warning messages (2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.widgets import Button



model_Type = input('Choose model type (FNN or CNN): ')
if model_Type == 'FNN':
    Activation = input('Choose activation function\n'+
                       'When using the sigmoid acivation, remove the normalisation of the data and see how the results'+
                       'differ from the normalised data\n'+
                       'Activation type: (sigmoid, relu or None): ')
    if Activation == 'None':
        Activation = None
        
print('\nLoad MNIST data\nDivide data in training and validation data sets\n')

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Set training parameters
batch_size = 128 
epochs     = 10
learning_rate = 0.001
Optimizer = 'Adam' # For instance Adam or SGD (change your learning rate accordinly)


# the data, split between train and test (validation) sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data normalization Scale images to the [0, 1] range 
# (For interest, try FNN model without normalization)
# Or try FNN with and without the sigmoid activation function
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255

# Make sure images have shape (N, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples\n")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)


################################ (Architecture) ###############################

# Note this only creates an overview of the model
# The model must be compiled before it can be used
def FFNModel(Activation='sigmoid'):
    """Create FNN model for MNIST
    Note every layer has a unique name. 
    This is usefull when a specific training strategy is required
    Or when we want to store the trained model and load it later
    For Fun replace the activation function sigmoid with None and
    see how the accuracy changes or with relu.
    
    Note sigmoid activation expects a normalized input, as range sigmoid [0-1] 
    If non-normalized data is used than all outputs of the first layer will have
    an activation of nearly 1 or neraly zero.
     
    
    Created model:
        Fully connected neural network with three dense layers
        First layer:  10 neurons, activation Activation (input)
        Second layer: 20 neurons, activation Activation (input)
        Third layer:  15 neurons, activation Activation (input)
        Output layer: 10 neurons, activation softmax
        
    This is a random created model, not much thought is given to the number of
    layers or the number of neurons per layer"""
    inputs = keras.Input(shape=input_shape,name='input')
    x = layers.Flatten(name='Flatten')(inputs)
    x = layers.Dense(10,activation=Activation,name='Dense_10_0')(x)
    x = layers.Dense(20,activation=Activation,name='Dense_20_1')(x)
    x = layers.Dense(15,activation=Activation,name='Dense_15_2')(x)
    outputs = layers.Dense(num_classes, activation="softmax",name='Output_10')(x)
    return inputs,outputs


def CCNModel():
    """Create a convolutional neural network for MNIST
    Note every layer has a unique name. 
    This is usefull when a specific training strategy is required
    Or when we want to store the trained model and load it later
    
    Model copied from Keras website
    
    Created model:
        Convolutional neural network
        First layer: kernel 3 x 3 (32 kernels)
        Second layer: Max pooling kernel 2 x 2
        Third layer: kernel 3 x 3 (64 kernels)
        Fourth layer: Max pooling kernel 2 x 2
        Output layer: 10 neurons, activation softmax
        
        Regularization: Dropout, rate 0.5"""
    inputs = keras.Input(shape=input_shape,name='input')
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu",name='Con_32_3x3_0')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2),name='MaxPool_2x2_0')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu",name='Con_64_3x3_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2),name='MaxPool_2x2_1')(x)
    x = layers.Flatten(name='Flatten')(x)
    x = layers.Dropout(0.5,name='Dropout_0.5_0')(x)
    outputs = layers.Dense(num_classes, activation="softmax",name='Output_10')(x)
    return inputs,outputs

# Callback for logging training info
# the on_epoch_end method will be caleld at the end of each epoch
class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        self.reset()

    def reset(self):
        self.trainloss        = []
        self.valloss          = []
        self.trainAccuracy    = []
        self.ValAccuracy      = []
        
    def on_epoch_end(self, epoch, logs=None):
        """Log loss and accuracy of the training and validation data sets"""
        self.trainloss.append(logs["loss"])
        self.trainAccuracy.append(logs['accuracy'])
        self.valloss.append(logs["val_loss"])
        self.ValAccuracy.append(logs['val_accuracy'])
        
    def get_data(self):
        return self.trainloss, self.valloss, self.trainAccuracy, self.ValAccuracy


MyCalBack = CustomCallback()


################################ (Create model) ###############################

# Create model
if model_Type == 'CNN':
    inputs,outputs= CCNModel()    
elif model_Type == 'FNN':
    inputs,outputs= FFNModel(Activation) 
else:
    raise Exception('Model_type must be FNN or CNN')
model = keras.Model(inputs=inputs, outputs=outputs, name='MNIST_Model')

# compile model. Note, a Graph is created that will run an a GPU
# When training all data and operations are on the GPU. Python is not used to
# exceute any operation
opt = getattr(keras.optimizers, Optimizer)(learning_rate=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Print overview of the model
print('Model overview')
for ii, layer in enumerate(model.layers):
    print('Layer '+str(ii)+'. Name: '+layer.name)
print('')
model.summary()


################################ (Training) ###################################


# Train model
print('\nStart training')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[MyCalBack])
trainloss,valloss,trainAccuracy,ValAccuracy = MyCalBack.get_data()


################################# (Plotting results) ##########################



# statistics
plt.close('Results')
fig1, (ax1,ax2) = plt.subplots(2,num='Results')
ax1.plot(range(epochs),trainloss,marker='o')
ax1.plot(range(epochs),valloss,marker='o')
ax1.set(xlabel='Epoch', ylabel='Loss')
ax1.set_title('Loss')
ax1.legend(['Training','Validation'])

ax2.plot(range(epochs),trainAccuracy,marker='o')
ax2.plot(range(epochs),ValAccuracy,marker='o')
ax2.set(xlabel='Epoch', ylabel='Accuracy')
ax2.set_title('Accuracy')
ax2.legend(['Training','Validation'])
fig1.show()
fig1.canvas.draw()
fig1.canvas.flush_events()  



# Create GUI to show Individual results
class ShowPrediction(object):
    
    def  __init__(self,data,label,model):
        plt.close('MNIST')
        
        # Init Figure
        self.fig, self.ax = plt.subplots(1,num='MNIST')
        self.ax.axis('off')
        self.fig.subplots_adjust(bottom=0.2)
        self.fig_msg = self.fig.text(0, 0.01,'Index: 0')
        
        # Init buttons
        self.axprev = plt.axes([0.7, 0.09, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.09, 0.1, 0.075])
        self.axwrong = plt.axes([0.7, 0.01, 0.21, 0.075])
        self.bnext  = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next1)
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev)
        self.bwrong  = Button(self.axwrong, 'Show wrong')
        self.bwrong.on_clicked(self.Show_Wrong_prediction)
        
        # Store data, init counter, init model
        self.ind   = -1
        self.data  = data
        self.label = label
        self.numimages = len(self.label[:,0])
        self.model = model
        self.batchsize = 128
        
        # Show first image
        self.Imhandle = self.ax.imshow(np.reshape(self.data[0,:,:],(28,28)),cmap='gray')
        self.next1('')
        plt.show()

    def predict(self): 
        """Inference on a single image
        Returns:
            label, score, GT label """
        out = self.model(np.expand_dims(self.data[self.ind,:,:],axis=0),training=False)[0]
        label = np.argmax(out)
        return label, out[label], np.argmax(self.label[self.ind,:])
    
    def predict_batch(self,batchsize):
        """Inference on a batch of images
        Returns:
            label, score, GT label """
        if self.ind >= (self.numimages-1):
            self.ind = 0
        index = min(self.ind+batchsize,self.numimages)
        out = self.model.predict(self.data[self.ind:index,:,:],batchsize)
        indeces = np.argmax(out,axis=1)
        GT_Labels   = np.argmax(self.label[self.ind:index,:],axis=1)
        self.ind = min(index, (self.numimages-1))
        scores = np.zeros(batchsize, dtype=out.dtype)
        for ii in range(out.shape[0]):
            scores[ii] =  out[ii,indeces[ii]]             
        return indeces, scores, GT_Labels
        
    
    def next1(self,event):
        """Show the next image, run a detection on it and show result"""

        self.ind = (self.ind + 1) % self.numimages
        self.Set_data_plot(np.reshape(self.data[self.ind,:,:],(28,28)), self.predict())
        self.Update_fig()
        
    def Show_Wrong_prediction(self,First = True):
        """Show the next prediction which is rong 
        (i.e. ground truth != predicted label)"""
        self.ind += 1
        while True:
            Index = self.ind
            label, score ,GT = self.predict_batch(self.batchsize)
            result = label != GT
            idx =  np.argmax(result)
            if result[idx]:
                self.ind = idx + Index
                self.Set_data_plot(np.reshape(self.data[self.ind,:,:],(28,28)), (label[idx], score[idx], GT[idx]))
                self.Update_fig()
                break
            if self.ind < Index:
                if First:
                    self.Show_Wrong_prediction(First = False)
                break
    
    def Set_data_plot(self,Image,Result):
        self.Imhandle.set_data(np.reshape(Image,(28,28)))
        self.ax.set_title('Predicted number: %1d, Score: %0.5f\n GT label: %1d' % (Result)) 
    
    def Update_fig(self):
        """Display index and update canvas"""
        self.Update_Index_text()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def Update_Index_text(self):
        self.fig_msg.set_text('Image: %d out of %d' % (self.ind+1, self.numimages))
            
    def prev(self,event):
        """Show previous detection
        (Runs the detection again on the previous number"""
        if self.ind > 0:
            self.ind -= 1 
        else:
            self.ind = self.numimages -1
        self.Set_data_plot(np.reshape(self.data[self.ind,:,:],(28,28)), self.predict())
        self.Update_fig()
        
GUI = ShowPrediction(x_test,y_test,model)
 
















