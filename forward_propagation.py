#Forward propagation model consisting of three sections
# linear_forward: basic linear module that computes Z and cache
# linear_activation_forward: forward propagation for the linear to activation layer
#   A_prev is input from previous layer
# L_model_forward: replicates the linear_activation_forward function L times
#   Forward prop for linear to relu (range) and linear to sigmoid (one iteration)

import numpy as np
from nn_functions import sigmoid, relu

def linear_forward(A,W,b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b'+str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation ="sigmoid")
    caches.append(cache)
    return AL, caches