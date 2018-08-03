import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
        	dict = cPickle.load(fo)
    	return dict

def cifar_10_data(file_addr, n):
	# n is number of data you want
	x_train = unpickle(file_addr)['data'][0:n,:]
	y_train = unpickle(file_addr)['labels'][0:n]
		
	return x_train, y_train
