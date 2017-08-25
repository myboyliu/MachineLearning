'''
AlexNet架构，5个卷积层 + 3个全连接层，具体分法
输入层
卷积层1(X*X,stride=Y,96个) + LRN层1 + 池化层1(3*3,stride=2,max_pool)
卷积层2(5*5,stride=1,256个) + LRN层2 + 池化层2(3*3,stride=2,max_poool)
卷积层3(3*3,stride=1,384个)
卷积层4(3*3,stride=1,384个)
卷积层5(3*3,stride=1,256个) + 池化层3(3*3,stride=2,max_poool)
非线性全连接层1(4096)
非线性全连接层2(4096)
线性全连接层3
softmax

变化的就是输入层，卷积层1以及线性全连接层
对于cifar数据集，因为它们都是32*32*3的图片，所以输入为32*32大小的，那么卷积层1的尺寸应该是5*5，步长stride=1
对于imagenet数据集，它们都是224*224*3的图片，所以输入为224*224大小的，卷积层1的尺寸应该是11*11，步长stride=1
对于mnist数据集，图片大小是28*28的，应该与cifar相同的配置

至于线性全连接层的输出，需要看最终有多少个分类来决定

在全连接层可以增加dropout
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
training_epochs = 2
display_step = 10
def WeightsVariable(shape, stddev, name):
    init_value = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=init_value, dtype=tf.float32, name=name)

def BiasesVariable(shape, init_value, name):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name)

def Conv2d(x,W,b,stride,padding,activation):
    y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    y = tf.nn.bias_add(y, b)
    y = activation(y)
    return y

def FullConnected(x, W, b, activation):
    y = tf.matmul(x, W)
    y = tf.add(y, b)
    y = activation(y)
    return y

def EvaluateModelOnDataset(sess, images, labels):
    n_samples = images.shape[0]
    per_batch_size = 100
    loss = 0
    acc = 0

    if (n_samples <= per_batch_size): #样本比较少，一次评估
        batch_count = 1
        loss, acc = sess.run([cross_entropy, accuracy],
                             feed_dict={images_holder : images, labels_holder : labels})
    else: #样本比较大，分批次评估
        batch_count = int(n_samples / per_batch_size)
        batch_start = 0
        for idx in range(batch_count):
            batch_loss, batch_acc = sess.run([cross_entropy, accuracy],
                                             feed_dict={images_holder : images[batch_start:batch_start + per_batch_size, :],
                                                        labels_holder: labels[batch_start:batch_start + per_batch_size, :]})
            batch_start += per_batch_size
            loss += batch_loss
            acc += batch_acc

    return loss / batch_count, acc / batch_count
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

with tf.name_scope('Inputs'):
    images_holder = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32)
    labels_holder = tf.placeholder(shape=[batch_size, 10], dtype=tf.int32)
    x_images = tf.reshape(images_holder, [-1, 28, 28, 1]) # mnist是单通道图片

with tf.name_scope('Inference'):
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[5,5,1,96], stddev=1e-1, name='weights1')
        biases = BiasesVariable(shape=[96], name='biases1', init_value=0.0)
        conv1_out = Conv2d(x_images, weights, biases, stride=1, padding='SAME', activation=tf.nn.relu)
    with tf.name_scope('LRN_1'):
        lrn1_out = tf.nn.lrn(conv1_out, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='lrn1')
    with tf.name_scope('Pool_1'):
        pool1_out = tf.nn.max_pool(lrn1_out, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Conv2d_2'):
        weights = WeightsVariable(shape=[5,5,96,256], stddev=1e-1, name='weights2')
        biases = BiasesVariable(shape=[256], name='biases2', init_value=0.0)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME', activation=tf.nn.relu)
    with tf.name_scope('LRN_2'):
        lrn2_out = tf.nn.lrn(conv2_out, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='lrn2')
    with tf.name_scope('Pool_2'):
        pool2_out = tf.nn.max_pool(lrn2_out, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Conv2d_3'):
        weights = WeightsVariable(shape=[3,3,256,384], stddev=1e-1, name='weights3')
        biases = BiasesVariable(shape=[384], name='biases3', init_value=0.0)
        conv3_out = Conv2d(pool2_out, weights,biases, stride=1, padding='SAME', activation=tf.nn.relu)
    with tf.name_scope('Conv2d_4'):
        weights = WeightsVariable(shape=[3,3,384,384], stddev=1e-1, name='weights4')
        biases = BiasesVariable(shape=[384], name='biases4', init_value=0.0)
        conv4_out = Conv2d(conv3_out, weights,biases, stride=1, padding='SAME', activation=tf.nn.relu)
    with tf.name_scope('Conv2d_5'):
        weights = WeightsVariable(shape=[3,3,384,256], stddev=1e-1, name='weights5')
        biases = BiasesVariable(shape=[256], name='biases5', init_value=0.0)
        conv5_out = Conv2d(conv4_out, weights,biases, stride=1, padding='SAME', activation=tf.nn.relu)
    with tf.name_scope('Pool_3'):
        pool3_out = tf.nn.max_pool(conv5_out, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    print_activations(pool3_out)
    # 特征提取完毕，转化特征
    with tf.name_scope('FeaturesRshape'):
        features = tf.reshape(pool3_out, shape=[batch_size, -1])
        print_activations(features)
        feats_dim = features.get_shape()[1].value
        print(feats_dim)
    with tf.name_scope('FC_nonlinear1'):
        weights = WeightsVariable(shape=[feats_dim, 4096], stddev=0.04, name='weights6')
        biases = BiasesVariable(shape=[4096], name='biases6', init_value=0.1)
        fc1_out = FullConnected(features, weights, biases, activation=tf.nn.relu)
    with tf.name_scope('FC_nonlinear2'):
        weights = WeightsVariable(shape=[4096, 4096], stddev=0.04, name='weights7')
        biases = BiasesVariable(shape=[4096], name='biases7', init_value=0.1)
        fc2_out = FullConnected(fc1_out, weights, biases, activation=tf.nn.relu)
    with tf.name_scope('FC_linear'):
        weights = WeightsVariable(shape=[4096, 10], stddev=1.0/4096, name='weights8')
        biases = BiasesVariable(shape=[10], name='biases8', init_value=0.1)
        logits = FullConnected(fc2_out, weights, biases, activation=tf.identity)

with tf.name_scope('Loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_holder)
    total_loss = tf.reduce_mean(cross_entropy)

with tf.name_scope('Train'):
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False, dtype=tf.int64)
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(total_loss, global_step=global_step)

with tf.name_scope('Evaulate'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_holder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.global_variables_initializer()
mnist = input_data.read_data_sets('../Total_Data/MNIST_data/', one_hot=True)

with tf.Session() as sess:
    sess.run(init_op)
    total_batches = int(mnist.train.num_examples / batch_size)
    print("Per batch Size: ", batch_size)
    print("Train sample Count: ", mnist.train.num_examples)
    print("Total batch Count: ", total_batches)

    training_step = 0
    for epoch in range(training_epochs):
        for batch_idx in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run([train_op], feed_dict={images_holder : batch_x,
                                         labels_holder : batch_y})

            training_step = sess.run(global_step) #跟上一句的效果是一样的
            # print(str(training_step) + ", " + str(display_step))
            if training_step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={images_holder: batch_x, labels_holder: batch_y})
                loss = sess.run(total_loss, feed_dict={images_holder: batch_x, labels_holder: batch_y})
                print("Training Epoch:"+ str(epoch) +", Training Step: " + str(training_step) +
                      ", Training Loss= " + "{:.6f}".format(loss) +
                      ", Training Accuracy= " + "{:.5f}".format(acc))

    print("训练完毕")
    test_samples_count = mnist.test.num_examples
    test_loss, test_accuracy = EvaluateModelOnDataset(sess, mnist.test.images, mnist.test.labels)
    print("Testing Samples Count:", test_samples_count)
    print("Testing Loss:", np.mean(test_loss))
    print("Testing Accuracy:", test_accuracy)