import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

w = tf.get_variable("w", initializer = tf.constant(20))
assign_w = w.assign(100)

with tf.Session() as sess:
	sess.run(w.initializer)
	sess.run(assign_w)
	print w.eval()
