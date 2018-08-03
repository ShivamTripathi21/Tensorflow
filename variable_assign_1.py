import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

var = tf.get_variable("var", initializer=tf.constant(2))
assign = var.assign(2 * var)

with tf.Session() as sess:
	sess.run(var.initializer)
	print var.eval()
	sess.run(assign)
	print var.eval()
	sess.run(assign)
	print var.eval()
	sess.run(assign)
	print var.eval()
