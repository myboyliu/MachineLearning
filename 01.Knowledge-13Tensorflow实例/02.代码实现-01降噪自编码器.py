'''
实现降噪自编码器
'''
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, n_transfer = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = n_transfer
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = dict()

        with tf.name_scope('RawInput'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])

        with tf.name_scope('NosieAdder'):
            self.scale = tf.placeholder(tf.float32)#噪声的方差尺度
            self.noise_x = self.x + self.scale * tf.random_normal((self.n_input,))# 加噪声

        with tf.name_scope('Encoder'):
            #使用Xavier进行初始化
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight1')
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='bias1')
            self.hidden = n_transfer(tf.add(tf.matmul(self.noise_x, self.weights['w1']), self.weights['b1'])) # 隐藏层输出

        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32), name='weight2')
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='bias2')
            self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        with tf.name_scope('Loss'):
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))

        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x : X, self.scale:self.training_scale})
        return cost
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict={self.x : X, self.scale : self.training_scale})
    def transform(self,X):
        return self.sess.run(self.hidden, feed_dict={self.x : X, self.scale : self.training_scale})
    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden : hidden})
    def reconstruction(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x : X, self.scale : self.training_scale})
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

#0均值，标准差为1
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

mnist = input_data.read_data_sets('../MNIST_data/', one_hot = True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
AGN_AC = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=200, n_transfer=tf.nn.softplus,
                                          optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                          scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = AGN_AC.partial_fit(batch_xs)
        avg_cost += cost / batch_size
    avg_cost /= total_batch

    if epoch % display_step == 0:
        print("epoch : %04d, cost = %.9f" % (epoch + 1, avg_cost))

print("Total cost : ", str(AGN_AC.calc_total_cost(X_test)))
# print('把计算图写入事件文件，在TensorBoard里面查看')
# writer = tf.summary.FileWriter(logdir='../logs', graph=AGN_AC.sess.graph)
# writer.close()