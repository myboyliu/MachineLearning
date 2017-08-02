import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1)) # 不要初始化为0
b1 = tf.Variable(tf.zeros([300]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([300, 300], stddev=0.1)) # 不要初始化为0
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 100], stddev=0.1)) # 不要初始化为0
b3 = tf.Variable(tf.zeros([100]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1)) # 不要初始化为0
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# loss = tf.reduce_mean(tf.square(y - prediction))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                              logits=prediction)) #交叉熵明显收敛更快，而且运行速度也快
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x : batch_xs, y : batchys, keep_prob:0.7})

        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,
                                                 y:mnist.test.labels,
                                                 keep_prob:0.7})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images,
                                                  y:mnist.train .labels,
                                                  keep_prob:0.7})
        print("Iter" + str(epoch) + ", Testing Accurancy " + str(test_acc) + ", Training Accurancy " + str(train_acc))


    print("Done")