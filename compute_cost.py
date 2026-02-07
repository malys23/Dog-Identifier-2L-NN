#Compute entropy cost J to check if model is actually learning

import numpy as np

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y))
    cost = np.squeeze(cost)
    return cost