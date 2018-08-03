import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

my_var = tf.Variable(10)

with tf.Session() as sess:
	sess.run(my_var.initializer)
	print my_var.eval()
	# increment by 10
	sess.run(my_var.assign_add(10)) # >> 20
	print my_var.eval()
	# decrement by 2
	sess.run(my_var.assign_sub(2)) # >> 18
	print my_var.eval()
	

