'''
使用mnist数据集，
输入层数据：data = 28 * 18 * 1
卷积层：conv2d = 5 * 5 * 1
输出输出：data = 24 * 24 * K(K就是卷积核的个数)
激活层ReLU
输出数据：data = 24 * 24 * K
池化层：pool2d - MaxPool
全连接层 units = 10
输出数据(特征):logits = 1 * 1 * 10(最终需要分10类：0-9)
Softmax：计算属于每个分类的概率
'''
import tensorflow as tf
import os
import csv
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

n_input = 784
n_classes = 10

def WeightsVariable(shape, name_str, stddev = 0.1):
    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    # initial = tf.truncated_normal(shape, stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str, stddev=0.00001):
    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    # initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name = name_str)
# 卷积层不做降采样
def Conv2d(x, W, b, stride=1, padding='SAME'):
    with tf.name_scope('Wx_b'):
       y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
       y = tf.nn.bias_add(y, b)
    return y

def Activation(x, activation=tf.nn.relu, name='relu'):
    with tf.name_scope(name):
        y = activation(x)
    return y

def Pool2d(x, pool = tf.nn.max_pool, k =2, stride=2):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID')

def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)

    return y

if __name__ == '__main__':
    with tf.Graph().as_default():
        # 输入
        with tf.name_scope('Inputs'):
            X_origin = tf.placeholder(tf.float32, [None, n_input], name='X_origin')
            Y_true = tf.placeholder(tf.float32, [None, n_input], name='Y_true')
            X_image = tf.reshape(X_origin, [-1,28,28,1])

        #前向推断
        with tf.name_scope('Inference'):
            with tf.name_scope('Conv2d'): # 卷积层
                weights = WeightsVariable(shape=[5,5,1,16], name_str='weights')
                biases = BiasesVariable(shape=[16], name_str='biases')
                conv_out = Conv2d(X_image, weights, biases, stride=1, padding='VALID')

            with tf.name_scope('Activate'):# 非线性激活层
                activate_out = Activation(conv_out, activation=tf.nn.relu,name='relu')

            with tf.name_scope('Pool2d'): #池化层
                pool_out = Pool2d(activate_out, pool=tf.nn.max_pool, k=2, stride=2)

            with tf.name_scope('FeatsReshape'): #将二维特征图变为一维特征向量，得到的是16个特征图，每个特征图是12*12的
                features = tf.reshape(pool_out, [-1, 12 * 12 * 16])

            with tf.name_scope('FC_Linear'): #全连接层
                weights = WeightsVariable(shape=[12 * 12 * 16, n_classes], name_str='weights')
                biases = BiasesVariable(shape=[n_classes], name_str='biases')
                Ypred_logits = FullyConnected(features, weights, biases,
                                              activate=tf.identity, # 恒等映射，没有经过激活函数，所以没有任何改变
                                              act_name='identity')

        #定义损失层
        with tf.name_scope('Loss'):
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=Y_true, logits=Ypred_logits
            ))

        #定义优化训练层
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            '''
            两个步骤：
            1.反向传播计算梯度
            2.利用梯度下降算法优化权重与偏置
            '''
            trainer = optimizer.minimize(cross_entropy_loss)

        #定义模型评估层
        with tf.name_scope('Evaluate'):
            correct_pred = tf.equal(tf.argmax(Ypred_logits, 1), tf.argmax(Y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        # summary_writer = tf.summary.FileWriter(logdir='../logs/05/', graph=tf.get_default_graph())
        # summary_writer.close()
        mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
        result_list = list()
        result_list.append(['learning_rate', learning_rate,
                            'training_epochs', training_epochs,
                            'batch_size', batch_size,
                            'display_step', display_step])
        result_list.append(['train_step', 'train_loss', 'validation_loss',
                            'train_step', 'train_accuracy', 'validation_accuracy'])
        with tf.Session() as sess:
            sess.run(init)
            total_batches = int(mnist.train.num_examples / batch_size)
            print("Per batch Size: ", batch_size)
            print("Train sample Count: ", mnist.train.num_examples)
            print("Total batch Count: ", total_batches)

            training_step = 0
            for epoch in range(training_epochs):
                for batch_idx in range(total_batches):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    sess.run(trainer, feed_dict={X_origin : batch_x,
                                                 Y_true : batch_y,
                                                 learning_rate : learning_rate_init})
                    training_step += 1
                    if training_step % display_step == 0:
                        start_idx = max(0, (batch_idx - display_step) * batch_size)
                        edit_idx = batch_idx * batch_size
                        train_loss, train_acc =EvaluteModelOnDataset()


