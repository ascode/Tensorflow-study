import tensorflow as tf

a = tf.constant([1,2,3])
b = tf.constant([4,5,6])

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        print(sess.run(a + b))
