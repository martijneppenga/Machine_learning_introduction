# Example of machine learning using MNIST

This directory contains same examples on how to create a machine learning program

## MnistKeras directry
This directory contains files to create a ML program to regonize handwritten numbers from the MNISt data base
The programs are created using TensorFlow. Each files contain examples on how to create a ML program. Furthermore some suggestions are given in the comments of each file on which hyperparamters to change and how they will influence the ML perforamce. The files are created to "play" with, so it is encouraged to make adjustemetns yourself and try out how it will influence the perfomance of the ML algorithm. 

MnistKerasConv.py file is an example on how to create, train and evaluate a convolutional or a fully connected neural network.
CustomTraining.py file creates custom training loops, data loaders and M.L. layers. (before using this file first inspect the MnistKerasConv.py, as the CustomTraining.py is an extension)


### Requirments
To use these files Tensorflow, matplotlib and numpy must be installed
It is advised to create a new enviroment and first intsall tensorfloww (i.e. use conda create --name myenv)
Before you install tensorflow make sure that visual studio is installded, cuda and cudnn


## Nielson directory
The Nielson directory contains a machine learning program created with numpy inorder to classify images using the MNISt data base. This directory is created to get an understanding on the arithmetic rules designed to create ML programs. The info.txt file in this directory contains more inforamtion on how to use them
