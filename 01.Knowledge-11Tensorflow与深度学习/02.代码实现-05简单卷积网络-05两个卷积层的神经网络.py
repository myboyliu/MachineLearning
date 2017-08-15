#卷积神经网络算法
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
mnist = input_data.read_data_sets("../Total_Data/MNIST_data/", one_hot=True)
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1,28,28,1]) # 大小是28*28的图片，通道数是1

'''
生成截断正态分布
[5,5,1,32]， 5*5是采样窗口，1是输入的通道数，32代表输出的通道数，也就是输出32个卷积核
'''
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))

'''
每一个卷积核一个偏置值，这里有32个卷积核，就有32个偏置值
'''
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
'''
strides的0，3位置都是1，默认的，第1个位置是x方向的步长，第2个位置是y方向的步长
2维卷积操作
'''
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
'''
最大池化
ksize的0，3位置都是1，默认的，第1个位置是x方向的大小，第2个位置是y方向的大小
'''
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

'''
28*28的图片，经过第一次卷积后还是28*28，这是因为卷积用的SAME padding，第一次池化后变为14*14因为池化步长为2
第二次卷积后为14*14，因为输入为14*14，卷积用的SAME padding，第二次池化后变成7*7，因为池化步长为2
经过上面操作后得到了64张7*7的平面
'''
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))#上一层有7*7*64=3136个神经元，全连接第一层有共1024个神经元
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))#1024个神经元，所以有1024个偏置

'''
把上一层的输出扁平化处理，也就是转化成1维
'''
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

'''
第二个连阶层
'''
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entroy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x : batch_xs, y : batchys, keep_prob:0.7})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,
                                            y:mnist.test.labels,
                                            keep_prob:0.7})
        print("Iter" + str(epoch) + ", Testing Accurancy " + str(acc) )

    print("Done")
