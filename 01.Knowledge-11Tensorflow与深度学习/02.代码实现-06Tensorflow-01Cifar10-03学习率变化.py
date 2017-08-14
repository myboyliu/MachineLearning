import tensorflow as tf
import os
import numpy as np
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

training_epochs = 5
num_examples_per_epoch_for_train = 10000
batch_size = 100
learning_rate_init = 0.1
learning_rate_final = 0.001
learning_rate_decay_rate = 0.5
num_batches_per_epoch = int(num_examples_per_epoch_for_train / batch_size)
num_epochs_per_decay = 1 # 每次过多少个epoch，学习率就会降低
learning_rate_decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

with tf.Graph().as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, learning_rate_decay_steps, learning_rate_decay_rate, staircase=False)
    # learning_rate = tf.train.polynomial_decay(learning_rate_init, global_step, learning_rate_decay_steps, learning_rate_final, power=0.5,cycle=False)
    # learning_rate = tf.train.natural_exp_decay(learning_rate_init, global_step, learning_rate_decay_steps, learning_rate_decay_rate, staircase=False)
    # learning_rate = tf.train.inverse_time_decay(learning_rate_init, global_step, learning_rate_decay_steps, learning_rate_decay_rate, staircase=False)
    weights = tf.Variable(tf.random_normal([9000, 9000], mean=0.0, stddev=1e9, dtype=tf.float32))
    myloss = tf.nn.l2_loss(weights, name='L2Loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(myloss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    results_list = list()
    results_list.append(['train_step', 'learning_rate', 'train_step', 'train_loss'])
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(training_epochs):
            print('***************************')
            for batch_idx in range(num_batches_per_epoch):
                current_learning_rate = sess.run(learning_rate)
                _, loss_value, training_step = sess.run([training_op, myloss, global_step])
                print("Training Epoch: " + str(epoch) +
                      ", Training Step: " + str(training_step) +
                      ", Learning Rate=" + "{:.6f}".format(current_learning_rate) +
                      ", Training_Loss=" + "{:.6f}".format(loss_value))
                results_list.append([training_step, current_learning_rate, training_step, loss_value])