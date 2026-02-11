#Backward propagation model consisting of three sections
# linear_backward: linear portion for one layer
# linear_activation_backward: backward propagation for the linear to activation layer
#   dA is post activation gradient for current layer
# L_model_backward: replicates the linear_activation_forward function L times
#   Backward prop for linear to relu (range) and linear to sigmoid (one iteration)

import numpy as np
from nn_functions import sigmoid_backward, relu_backward

# Back Prop for one layer
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot((1/m) * dZ, A_prev.T)
    db = (1/m) * np.dot(W.T, dZ)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

# Back Prop for (lienar to activation) layer
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation =="sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# Back Prop for (linear to relu)*(L-1) to (linear to sigmoid) model
## Initialize back propagation with dAL
## Then get the sigmoid to linear gradients
## Then loop backwards to get the relu to linear gradients 

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA"+str(L-1)] = dA_prev_temp
    grads["dW"+str(L)] = dW_temp
    grads["db"+str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    
    return grads