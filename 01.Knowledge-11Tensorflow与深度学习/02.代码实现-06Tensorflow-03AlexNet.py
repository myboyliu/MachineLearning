import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.01
training_epochs = 10
batch_size = 32
display_step = 10
conv1_kernel_num = 64
conv2_kernel_num = 192
conv3_kernel_num = 384
conv4_kernel_num = 256
conv5_kernel_num = 256
fc1_units_num = 4096
fc2_units_num = 4096

image_size = 224
image_channel = 3
n_classes = 1000

def WeightsVariable(shape, name_str, stddev=0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str, init_value = 0.0):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def Conv2d(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y

def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return pool(x, ksize=[1,k,k,1], strides=[1,stride,stride,1], padding=padding)

def FullyConnected(x, W, b, activate = tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)

    return y

def Interence(images_holder):
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[11,11,image_channel, conv1_kernel_num], name_str='weights',
                                  stddev=1e-1)
        biases = BiasesVariable(shape=[conv1_kernel_num], name_str='biases', init_value=0.9)
        conv1_out = Conv2d(images_holder, weights, biases, stride=4, padding='SAME')
    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')
    with tf.name_scope('Conv2d_2'):
        weights = WeightsVariable(shape=[5,5,conv1_kernel_num, conv2_kernel_num], name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv2_kernel_num], name_str='biases', init_value=0.9)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')

