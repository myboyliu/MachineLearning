import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import math

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NUM_CLASSES=10
IMAGE_SIZE=28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(Images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                                  stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(Images, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                  stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                  stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits, name='xentropy') # 将logits转化为概率分布
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss) #标量汇总
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False) # 迭代次数
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # 比较最大的一个概率跟实际比较，如果一样就成功，否则就失败；如果是2，那么就是找到最大的两个概率，只要有一个跟实际一样，那么就成功
    return tf.reduce_sum(tf.cast(correct, tf.int32))