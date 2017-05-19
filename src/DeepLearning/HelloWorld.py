import tensorflow as tf

hello = tf.constant("Hello World")
sess = tf.Session()
b = sess.run(hello)
print(b)

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print(sess.run(add, feed_dict={a : 2, b : 3}))