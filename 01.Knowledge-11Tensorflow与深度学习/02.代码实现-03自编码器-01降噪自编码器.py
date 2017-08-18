'''
实现降噪自编码器
'''
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N_INPUT = 784
N_HIDDEN = 200
N_SCALE = 0.01
LEARNING_RATE_INIT = 0.01

training_epochs = 20
batch_size = 128
display_step = 1

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#0均值，标准差为1
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

if __name__ == '__main__':
    with tf.name_scope("Inputs"):
        image_holder = tf.placeholder(tf.float32, [None, N_INPUT])
    with tf.name_scope('NosieAdder'):
        scale_holder = tf.placeholder(tf.float32)
        noise_x = image_holder + scale_holder * tf.random_normal((N_INPUT,))
    with tf.name_scope('Encoder'):
        weights1 = tf.Variable(xavier_init(N_INPUT, N_HIDDEN), name='weight1')
        bias1 = tf.Variable(tf.zeros([N_HIDDEN], dtype=tf.float32), name='bias1')
        hidden = tf.nn.softplus(tf.add(tf.matmul(noise_x, weights1), bias1))
    with tf.name_scope('Reconstruction'):
        weights2 = tf.Variable(tf.zeros([N_HIDDEN, N_INPUT], dtype=tf.float32), name='weight2')
        bias2 = tf.Variable(tf.zeros([N_INPUT], dtype=tf.float32), name='bias2')
        reconstruction = tf.add(tf.matmul(hidden, weights2), bias2)
    with tf.name_scope('Loss'):
        Loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, image_holder), 2))
    with tf.name_scope('Train'):
        Train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_INIT).minimize(loss=Loss)

    # print('把计算图写入事件文件，在TensorBoard里面查看')
    # writer = tf.summary.FileWriter(logdir='../logs', graph=AGN_AC.sess.graph)
    # writer.close()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        mnist = input_data.read_data_sets('../Total_Data/MNIST_data/', one_hot = True)
        X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
        n_samples = int(mnist.train.num_examples)

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)
                cost, opt = sess.run((Loss, Train), feed_dict={image_holder : batch_xs, scale_holder:N_SCALE})
                avg_cost += cost / batch_size
            avg_cost /= total_batch

            if epoch % display_step == 0:
                print("epoch : %04d, cost = %.9f" % (epoch + 1, avg_cost))
        totalLoss = sess.run(Loss, feed_dict={image_holder: X_test, scale_holder : N_SCALE})
        print("Total cost : ", str(totalLoss))



