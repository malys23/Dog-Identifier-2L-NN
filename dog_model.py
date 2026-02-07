import numpy as np
import h5py
import matplotlib.pyplot as plt
from nn_functions import sigmoid, sigmoid_backward, relu, relu_backward

from initialize_parameters import initialize_parameters
from initialize_parameters_deep import initialize_parameters_deep
from forward_propagation import linear_forward, linear_activation_forward, L_model_forward
from compute_cost import compute_cost
from backward_propagation import linear_backward, linear_activation_backward, L_model_backward
from update_parameters import update_parameters

train_set_x_orig = np.load('train_dogvnondog/train_set_x.npy')
train_set_y = np.load('train_dogvnondog/train_set_y.npy')
test_set_x_orig = np.load('test_dogvnondog/test_set_x.npy')
test_set_y = np.load('test_dogvnondog/test_set_y.npy')
classes = np.load('train_dogvnondog/list_classes.npy')

#1: Resizing and find values for m_train, m_test, and num_px
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_y = train_set_y.reshape(1, m_train)
test_y = test_set_y.reshape(1, m_test)

#2: Reshape data sets so images are flattened
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#3: Standardize dataset (divide by 255, max val of pixel channel)
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

