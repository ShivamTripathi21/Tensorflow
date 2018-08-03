##################################################
##################################################
##################################################
#######                                    #######  
####### TENSORFLOW STD. NEURAL NET. API    #######
####### CREATED BY :- SHIVAM TRIPATHI      #######
####### VERSION :- 1.0.0.0                 #######
####### SPEC :- Activation layer : RELU    #######
#######         Hidden layer : 50          #######
#######         CIFAR-10 Data set          #######
#######                                    #######
##################################################
##################################################
##################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import data_utils

file_addr = 'data/cifar-10/data_batch_1'
num_data = 20

#cifar_10_train_data(data_file_address, number_of_data_required)
#if want another data for testing or validation just change the file address

x,y = data_utils.cifar_10_data(file_addr,20)

# print np.shape(x),' ',np.shape(y)
# get x= (n X k) matrix and y = (n,) vector

#create varible for number of data, input size, hidden size and output size
n = tf.constant(num_data, name="num_of_data")
k = tf.constant(120*120, name="input_size")
hidden_size = tf.constant(50, name="hidden_size")
out_size = tf.constant(10, name="out_size")

# create random variabels for parameter
# hidden neural net size : 50
w1 = tf.get_variable("w1", initializer = tf.random_uniform([k, hidden_size])) 
b1 = tf.get_variable("b1", initializer = tf.zeros([hidden_size]))
w2 = tf.get_variable("w2", initializer = tf.random_uniform([hidden_size, out_size]))
b2 = tf.get_variable("b2", initializer = tf.zeros([out_size]))

# create place holder for variables
# here we have two variables one x:images of 32 x 32 pixel and another is y:lables of images

x, y = tf.placeholder(tf.float32, name="X"), tf.placeholder(tf.float32, name="Y")




