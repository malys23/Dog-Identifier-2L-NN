import numpy as np
import h5py
import matplotlib.pyplot as plt
from nn_functions import sigmoid, sigmoid_backward, relu, relu_backward

from initialize_parameters import initialize_parameters
from initialize_parameters_deep import initialize_parameters_deep
from forward_propagation import linear_forward, linear_activation_forward, L_model_forward
from compute_cost import compute_cost
from backward_propagation import linear_backward, linear_activation_backward, L_model_backward

train_set_x_orig = np.load('train_dogvnondog/train_set_x.npy')
train_set_y = np.load('train_dogvnondog/train_set_y.npy')
test_set_x_orig = np.load('test_dogvnondog/test_set_x.npy')
test_set_y = np.load('test_dogvnondog/test_set_y.npy')
classes = np.load('train_dogvnondog/list_classes.npy')
