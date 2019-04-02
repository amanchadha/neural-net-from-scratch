#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Aman Chadha"
__version__     = "1.1"
__maintainer__  = "Aman Chadha"
__email__       = "aman@amanchadha.com"
__website__     = "www.amanchadha.com"

"""
    Neural network from scratch to calculate the XOR function with:
    - one hidden layer 
    - number of hidden units as a hyperparameter
"""

import numpy as np

##################### HYPERPARAMETERS OF THE NEURAL NET #####################
nX              = 2  # No. of neurons in first layer
nH              = 2  # No. of neurons in hidden layer
nY              = 1  # No. of neurons in output layer
NUM_ITER        = 1000
LEARNING_RATE   = 0.3
#############################################################################
TEST_DATASET    = ((0, 0), (0, 1), (1, 0), (1, 1))
#############################################################################

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initializeParameters(nX, nH, nY):
    W1 = np.random.randn(nH, nX)
    b1 = np.zeros((nH, 1))
    W2 = np.random.randn(nY, nH)
    b2 = np.zeros((nY, 1))

    parameters = {
        "W1": W1,
        "b1" : b1,
        "W2": W2,
        "b2" : b2
    }

    return parameters

def forwardProp(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "A1": A1,
        "A2": A2
    }

    return A2, cache

def calculateCost(A2, Y, m):
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))/m
    cost = np.squeeze(cost)

    return cost

def backwardProp(X, Y, cache, parameters, m):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def updateParameters(parameters, gradients, learningRate):
    W1 = parameters["W1"] - learningRate*gradients["dW1"]
    b1 = parameters["b1"] - learningRate*gradients["db1"]
    W2 = parameters["W2"] - learningRate*gradients["dW2"]
    b2 = parameters["b2"] - learningRate*gradients["db2"]
    
    new_parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return new_parameters

def trainNeuralNet(X, Y, nX, nH, nY, numIter, learningRate, m):
    parameters = initializeParameters(nX, nH, nY)

    for i in range(0, numIter+1):
        a2, cache = forwardProp(X, parameters)

        cost = calculateCost(a2, Y, m)

        gradients = backwardProp(X, Y, cache, parameters, m)

        parameters = updateParameters(parameters, gradients, learningRate)

        if not i%100:
            print('Error/Loss/Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters

def predict(X, parameters):
    yHat, cache = forwardProp(X, parameters)
    yHat = np.squeeze(yHat)

    yPredict = 1 if yHat >= 0.5 else 0

    return yPredict

def testNeuralNet(trainedParameters):
    # Test 2x1 vector to calculate the XOR of its elements
    # Try (0, 0), (0, 1), (1, 0), (1, 1)
    for (i, y) in TEST_DATASET:
        XTest = np.array([[i], [y]])

        YPredict = predict(XTest, trainedParameters)

        print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(XTest[0][0], XTest[1][0], YPredict))
    
def main():
    np.random.seed(2)

    # Training examples by columns
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

    # The outputs of the XOR for every example in X
    Y = np.array([[0, 1, 1, 0]])

    # No. of training examples
    m = X.shape[1]

    print "*"*50
    print "Training Neural Net"
    print "*"*50
    trainedParameters = trainNeuralNet(X, Y, nX, nH, nY, NUM_ITER, LEARNING_RATE, m)

    print "*"*50
    print "Testing Neural Net"
    print "*"*50
    testNeuralNet(trainedParameters)

if __name__ == "__main__": main()