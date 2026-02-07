# Two Layer NN which follows the Linear to Relu to Linear to Sigmoid model
## Define number of examples and cost to store costs
## Initialize dictionary of parameters and call them
## Loop gradient descent to compute forward prop, compute cost, compute backward prop
##      then update parameters
## Returns parameters and costs

import numpy as np
from initialize_parameters import initialize_parameters
from forward_propagation import linear_activation_forward
from compute_cost import compute_cost
from backward_propagation import linear_activation_backward
from update_parameters import update_parameters

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims
    
    paramters = initialize_parameters(n_x, n_h, n_y)    
    W1 = paramters["W1"]
    b1 = paramters["b1"]
    W2 = paramters["W2"]
    b2 = paramters["b2"]
    
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation= "sigmoid")
        
        cost = compute_cost(A2, Y)
        
        dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        if print_cost and (i % 100 == 0 or i==num_iterations-1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
        
    return parameters, costs