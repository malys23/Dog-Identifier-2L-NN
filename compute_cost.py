import numpy as np

#Compute cost J to check i fmodel is actually learning
def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = (-1/m)*np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y))
    cost = np.squeeze(cost)
    return cost 