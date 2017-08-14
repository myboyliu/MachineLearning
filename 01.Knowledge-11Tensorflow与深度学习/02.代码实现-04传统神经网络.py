'''
Iter1, Testing Accurancy 0.9013, Training Accurancy 0.898982
Iter2, Testing Accurancy 0.9096, Training Accurancy 0.907982
Iter3, Testing Accurancy 0.917, Training Accurancy 0.914382
Iter4, Testing Accurancy 0.9215, Training Accurancy 0.921
Iter5, Testing Accurancy 0.9218, Training Accurancy 0.925309
Iter6, Testing Accurancy 0.9287, Training Accurancy 0.926727
Iter7, Testing Accurancy 0.9286, Training Accurancy 0.929982
Iter8, Testing Accurancy 0.9331, Training Accurancy 0.933945
Iter9, Testing Accurancy 0.9345, Training Accurancy 0.934255
Iter10, Testing Accurancy 0.9364, Training Accurancy 0.938182
Iter11, Testing Accurancy 0.9357, Training Accurancy 0.938145
Iter12, Testing Accurancy 0.9399, Training Accurancy 0.943364
Iter13, Testing Accurancy 0.9397, Training Accurancy 0.943382
Iter14, Testing Accurancy 0.9402, Training Accurancy 0.944673
Iter15, Testing Accurancy 0.9455, Training Accurancy 0.945655
Iter16, Testing Accurancy 0.9447, Training Accurancy 0.946691
Iter17, Testing Accurancy 0.9454, Training Accurancy 0.948855
Iter18, Testing Accurancy 0.9456, Training Accurancy 0.949745
Iter19, Testing Accurancy 0.9459, Training Accurancy 0.950436
Iter20, Testing Accurancy 0.9459, Training Accurancy 0.952455
Iter21, Testing Accurancy 0.9477, Training Accurancy 0.952073
Iter22, Testing Accurancy 0.9483, Training Accurancy 0.953636
Iter23, Testing Accurancy 0.9463, Training Accurancy 0.954545
Iter24, Testing Accurancy 0.9491, Training Accurancy 0.955218
Iter25, Testing Accurancy 0.949, Training Accurancy 0.955418
Iter26, Testing Accurancy 0.9513, Training Accurancy 0.955782
Iter27, Testing Accurancy 0.9465, Training Accurancy 0.955509
Iter28, Testing Accurancy 0.9537, Training Accurancy 0.9574
Iter29, Testing Accurancy 0.9537, Training Accurancy 0.958236
Iter30, Testing Accurancy 0.9523, Training Accurancy 0.959218
'''
'''传统神经网络算法'''
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size # //表示整数除法

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10]) # 最后的输出是0~9这10个数字，所以是10
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1)) # 不要初始化为0
b1 = tf.Variable(tf.zeros([300]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([300, 150], stddev=0.1)) # 不要初始化为0
b2 = tf.Variable(tf.zeros([150]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([150, 50], stddev=0.1)) # 不要初始化为0
b3 = tf.Variable(tf.zeros([50]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1)) # 不要初始化为0
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

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