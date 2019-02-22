import tensorflow as tf


temp = tf.range(0, 10)*10 + tf.constant(1, shape=[10])
temp2 = tf.gather(temp, [1, 2, 9])
with tf.Session() as sess:
    print(sess.run(temp))
    print(sess.run(temp2))
