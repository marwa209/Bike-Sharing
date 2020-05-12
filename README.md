# Bike-Sharing
Developed as coursework for Udacity "Deep Learning Fundamentals" Nanodegree. In this project, I built a neural network to predict daily bike rental ridership.
In this project, i made a neural network to solve a prediction problem in a real dataset! The Neural network was built from "scratch", using only NumPy to assist.By building a neural network from scratch, you will understand much better how gradient descent, backpropagation, and other important concepts of neural networks work. You also get a chance to see your network solving a real problem!


# Problem
Statement: Neural Network for predicting Bike Sharing Rides. HNeural network will predict how many bikes a company needs because if they have too few they are losing money from potential riders and if they have too many they are wasting money on bikes that are just sitting around. So Neural Network will predict from the hisitorical data how many bikes they will need in the near future.

Network Description: The network has two layers, a hidden layer and an output layer. The hidden layer uses the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is f(x)=xf(x)=x . A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called forward propagation. We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called backpropagation.

# Programming language:
All the code in runs in Python 3 without failing, and all unit tests pass


# Python libraries and prerequisites
* Python 3.7
* Numpy 
* Pandas 
* Matplotlib 
* Jupyter Notebook
* Torchvision
* PyTorch 

# Installing 
* conda install -c conda-forge numpy
* conda install -c conda-forge pandas
* conda install -c conda-forge matplotlib
* pip install torchvision
* conda install -c pytorch pytorch

# Tools:
TensorFlow


# Environment
Anaconda

# Project Structure
this projects run on jyputer notebook and the output is BikesSharing.ipynb which contains the output of running the code 
also building the neural network code is found the the file 


BikesSharing.ipynb - Main project file

my_answers.py the algorithm of the neural network structure

Bike-sharing-Dataset : This Bike-Sharing-Dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the cnt column. You can see the first few rows of the data above.

Bike-Sharing-Dataset/day.csv - Information about each trip taken using the bike share system by day.

Bike-Sharing-Dataset/hour.csv - Information about each trip taken using the bike share system by hour.

Bike-Sharing-Dataset/Readme.txt - General dataset description.
