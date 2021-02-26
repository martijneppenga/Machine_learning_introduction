# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:31:28 2020

@author: meppenga

Example on how to create a custom layer and custom training loop, and data
generator

This is an extension to the MnistKerasConv script

Two different models are created, one for training and one for inference

The weigths of the infernce model are loaded from the training model by first
saving the weigths of the training model, and then loading the saved weigths for
the inference model. If the weigth file already exist it will ask permission
to overwrite the file.
Note, if you don't want to save the weigth file, it will load the weights without
saving it

It is encouraged to play a bit with this script
Create your own layers, or loss function, add metrics to evaluate the trainig
results. etc.


"""
import os
# Set tensorflow logging verbosity
# Ignore all info and warning messages (2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.python.keras.layers as KE 
from matplotlib.widgets import Button
from tensorflow.keras.utils import Sequence


print('\nLoad MNIST data\nDivide data in training and validation data sets\n')

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Set training parameters
batch_size = 128 
epochs     = 10
# try also the learning rate  0.001, 0.005, 0.0005, 0.002, and 0.01 for adam
# for SGD try 2.0, 0.2, 0.02, and 10
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
# the last one in the shape is required as we going to use convolutional layers
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples\n")





################################ (Model custom layers) ########################
# This part of the script creates the architecture of the machine learning network
# Note that many of the examples here are more complicated then neccesary, as
# many build in modules exist in TensorFlow. However, when you want to create 
# more advanced M.L. modules it is usefull to know how you can create custom loss
# functions, training loops, and data generators. This gives you more control
# over the entire training and inference process while still be able to use many
# of the build in methods
#
# First a custom loss function is created by creating a custom layer
# Next an identity layer is created (google resnet for more info on these type
# of layers)
# Then a data generator is created and finally a custom model is created to train
# the model


# Custom loss function
class CustomLoss(KE.Layer):
    # Instead of KE.Layer you can also use layers.Layer as parent class
    # should not make a differnce, but I was used to use the KE.Layer class

    def __init__(self, Output_size, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.Output_size = Output_size

    def call(self, Prediction, GroundTruth):
        """Loss is the absolute loss value between predicted and ground truth
        value"""
        return tf.math.abs(Prediction-GroundTruth)

    def compute_output_shape(self, input_shape):
        return (None, self.Output_size)


class IdentityBlock(KE.Layer):
    # instead of KE.Layer you can also use layers.Layer as parent class
    # should not make a differnce, but I was used to use the KE.Layer class
    
    def __init__(self, Filters, _name_, **kwargs):
        """Applies an identity mapping block. It applies three convolutional 
        filters to the input and addes the input tensor to the output tensor
        of the three convolutional layers
        Note that the convolutional layers are build in the init method and
        not in the call method. A layer with weights must be build before the
        call method is called (other wise add a build method to build a layer)
        
        Input:
            Filters: tuple of ints, number of filters for each layer. Note the last
                filters must be the same size as the input tensor of this block
            name: str _name_ of the layer"""
        super(IdentityBlock, self).__init__(**kwargs)
        self.Filter1 = layers.Conv2D(Filters[0], padding='same', kernel_size=(3, 3), activation="relu",name=_name_+'_Con1_'+str(Filters[0])+'_3x3_1')
        self.Filter2 = layers.Conv2D(Filters[1], padding='same', kernel_size=(3, 3), activation="relu",name=_name_+'_Con2_'+str(Filters[1])+'_3x3_1')
        self.Filter3 = layers.Conv2D(Filters[2], padding='same', kernel_size=(3, 3), activation="relu",name=_name_+'_Con3_'+str(Filters[2])+'_3x3_1')
        self._name_ = _name_

        
    def call(self, x_in):
        """Creates the idenity mapping layer"""
        x = self.Filter1(x_in)
        x = self.Filter2(x)
        x = self.Filter3(x)
        x += x_in
        return layers.Activation('relu',name=self._name_+'_Relu')(x)


        

# Custom data generator
class Data_Generator(Sequence):
    
    def __init__(self, Batch_Size, xdata, ydata, num_classes):
        """Creates a data generator
        Note normally data would be loaded from disk instead from RAM, however
        as MNIST is a relative small data set it fits prefectlly in RAM so no
        need to create a fancy dataloader to load data from disk.
        We just load all data at once
        
        
        Data generators are generally used to limit the amount of data loaded
        into RAM. We however use it to control the data input for the custom 
        created training methods (i.e. see CustomModel class)
        Input:
            Batch_Size: int, batch size
            xdata: array like images to run the prediction on
            ydata: array like, ground truth
            num_classes: int, nuber of out put classes (should just be 10)
            """
        self.Batch_Size = Batch_Size
        self.xdata = xdata
        self.ydata = ydata
        self.num_classes = num_classes
        self.Index = np.arange(len(ydata))
        
     
    def on_epoch_end(self):
        """Shuffle data at the end of each epoch"""
        np.random.shuffle(self.Index)   
     
    def __len__(self):
        """Method to get the number of batches"""
        return (len(self.ydata) // self.Batch_Size)
    
    def __getitem__(self,idx):
        """Method to load the data. Note the input idx is the batch number
        Method has to return two outpus (i.e. input model and ground truth)
        Here we give the ground truth as an input to the model such that the 
        loss is callculated directly during training
        Therefore, the ground truth output is an empty list"""
        # get the indices of the data for a batch
        Index = self.Index[idx * self.Batch_Size: (idx + 1) * self.Batch_Size]
        # Create the data batch and the ground truth batch
        x_Batch = self.xdata[Index,:,:,:]
        y_Batch = keras.utils.to_categorical(self.ydata[Index], self.num_classes)
        return [x_Batch,y_Batch], []

# Custom training model
# as the loss value will be calculated within the model itslef by the loss layer
# a custom model is required to train the ML algorithm
# yet we want to use many of the methods in the keras model class, so we only
# create a new class with the keras model class as its parent and overwrite 
# the training method (train_step)  and the validation method (test_step)
# also the property metics is added such that the we can caculated some metrics
# during training and validation steps

# get the keras build-in metrics
loss_tracker = keras.metrics.Mean(name="loss")   
accuracy_tracker = keras.metrics.Accuracy(name='accuracy') 

# Create custom training class
class CustomModel(keras.Model):
    

    def train_step(self, data):
        """This method trains the model"""
        # Unpack data
        Inputs, Outputs = data
        
        # make a prediction for the data, Record operations for automatic 
        # differentiation using the gradient tape
        # note only put things in the resource manager (with.....) that require
        # automatic differentiation. Other pieces of code will slow down computation
        # signiicantly
        with tf.GradientTape() as tape:
            # Forward pass
            Predict, loss = self(Inputs, training=True) 
            # average over batch (not 100% sure if this is neccesary)
            loss = tf.reduce_mean(loss, keepdims=True)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights (only for trainable layers, i.e. CustomLoss layer is
        # not trainable for instance)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        accuracy_tracker.update_state(tf.math.argmax(Inputs[1],axis=1), tf.math.argmax(Predict,axis=1))

        # Return dict with loss and accuracy values
        return {"loss": loss_tracker.result(), "accuracy": accuracy_tracker.result()}
    


    def test_step(self, data):
        """Same as train step, but used for validation (so no logging and updating
        of the gradients"""
        # Unpack the data
        Inputs, Outputs = data
        # Compute predictions
        Predict, Loss = self(Inputs, training=False)
        loss = [tf.reduce_mean(Loss, keepdims=True)]
        # Updates the metrics tracking the loss
        loss_tracker.update_state(loss)
        accuracy_tracker.update_state(tf.math.argmax(Inputs[1],axis=1), tf.math.argmax(Predict,axis=1))
        # Return a dict mapping metric names to current value.
        return {"loss": loss_tracker.result(), "accuracy": accuracy_tracker.result()}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, accuracy_tracker]




################################ (Architecture) ########################


def CCNModel(mode, input_shape, num_classes):
    """Create a convolutional neural network for MNIST
    Note every layer has a unique name. 
    This is usefull when a specific training strategy is required
    Or when we want to store the trained model and load it later
    
    Model copied from Keras website
    Adjusted to add custom loss
    
    Input:
        mode: str (inference or training)
            inference mode: input is a batch of images, output is the predicted 
                label for each image
            training mode: input is batch of images and batch of ground truth
                output is predicted label and the loss value
    
    Created model:
        Convolutional neural network
        First layer: kernel 3 x 3 (32 kernels)
        Second layer: Max pooling kernel 2 x 2
        Third layer: Identity mapping (filters (5,8,32))
        Fourth layer: kernel 3 x 3 (64 kernels)
        Fifth layer: Max pooling kernel 2 x 2
        Output layer: 10 neurons, activation softmax
        
        Regularization: Dropout, rate 0.5"""
    if not (mode == 'inference' or mode =='training'):
       raise Exception('Mode must be either inference or training') 
        
    inputs = keras.Input(shape=input_shape,name='input')
    
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu",name='Con_32_3x3_0')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2),name='MaxPool_2x2_0')(x)
    x = IdentityBlock((5, 8, x.shape[-1]), 'Identity')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu",name='Con_64_3x3_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2),name='MaxPool_2x2_1')(x)
    x = layers.Flatten(name='Flatten')(x)
    x = layers.Dropout(0.5,name='Dropout_0.5_0')(x)
    if mode == 'inference':
        outputs = layers.Dense(num_classes, activation="softmax",name='Output_10')(x)
        return inputs, outputs
    elif mode == 'training':
        GT_input = keras.Input(shape=(num_classes),name='GT_input')
        outputs = layers.Dense(num_classes, activation="softmax",name='Output_10')(x)
        Loss_value = CustomLoss(num_classes)(outputs,GT_input)
        return [inputs, GT_input], [outputs, Loss_value]
    

# Callback for logging training info
# the on_epoch_end method will be caleld at the end of each epoch
# Note that this also could have been added into the custom model class
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





################################ (Create model) ###############################

# Create model
inputs, outputs = CCNModel('training', input_shape, num_classes)
model = CustomModel(inputs=inputs, outputs=outputs, name='MNIST_Model_training')

# compile model. Note, a Graph is created that will run an a GPU
# When training all data and operations are on the GPU. Python is not used to
# exceute any operation
opt = getattr(keras.optimizers, Optimizer)(learning_rate=learning_rate)
model.compile(optimizer=opt)



# Print overview of the model
print('Model overview')
for ii, layer in enumerate(model.layers):
    print('Layer '+str(ii)+'. Name: '+layer.name)
print('')
model.summary()


################################ (Training) ###################################

# init callbacks
MyCalBack = CustomCallback()

# Create data generators
Generator_train = Data_Generator(batch_size, x_train, y_train ,num_classes)
Generator_Val   = Data_Generator(batch_size, x_test, y_test, num_classes)

# Train model
print('\nStart training')
model.fit(Generator_train, validation_data=Generator_Val, epochs=epochs, callbacks=[MyCalBack])
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



class ShowPrediction(object):
    
    def  __init__(self,data,label,model,num_classes):
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
        self.numimages = len(self.label)
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
        return label, out[label], self.label[self.ind]
    
    def predict_batch(self,batchsize):
        """Inference on a batch of images
        Returns:
            label, score, GT label """
        if self.ind >= (self.numimages-1):
            self.ind = 0
        index = min(self.ind+batchsize,self.numimages)
        out = self.model.predict(self.data[self.ind:index,:,:],batchsize)
        indeces = np.argmax(out,axis=1)
        GT_Labels   = self.label[self.ind:index]
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
        


# Create inference model
inputs,outputs = CCNModel('inference', input_shape, num_classes)
model_Inference = CustomModel(inputs=inputs, outputs=outputs, name='MNIST_Model_inference')

# Compile model
opt = getattr(keras.optimizers, Optimizer)(learning_rate=learning_rate)
model_Inference.compile(optimizer=opt)  
  
# save and or load weights 
if 'y' == input('Do you want to save the weight file (y/n)? '):
    weight_file = os.path.join(os.path.split(__file__)[0],'MNIST.h5')
    # verify if you want to overwrite the weight file if it exist
    if os.path.isfile(weight_file):
        if 'y' != input('Save file already exist\n'+weight_file+'\nDo you want to overwrite this file (y/n)? '):
            model_Inference.set_weights(model.get_weights())
        else:
            # save the weigt file of the training model and load it for the inference model
            # Note that we use the by_name=True  when loading the weigths
            # Since the model only differ in its output and each layer has the same name
            # we can load the data simply by the name of the layer
            model.save_weights(weight_file, overwrite=True) 
            model_Inference.load_weights(weight_file, by_name=True)
    else:
        model.save_weights(weight_file, overwrite=True) 
        model_Inference.load_weights(weight_file, by_name=True)
else:
    model_Inference.set_weights(model.get_weights())

# Create Gui to show results
GUI = ShowPrediction(x_test,y_test,model_Inference,num_classes)


    



















