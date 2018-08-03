import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
w = tf.get_variable("big_matrix", initializer=tf.random_uniform([700,10]))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print s.eval(),'\n',m.eval(),'\n',w.eval(),'\n'

