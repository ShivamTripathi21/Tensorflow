import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.placeholder(tf.int32, shape=[3])
b = tf.constant([5, 5, 5], tf.int32)

c = a + b # tf.add(a, b)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
	print sess.run(c, feed_dict={a: [1, 3, 4]})
