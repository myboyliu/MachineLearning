import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# 784 -> 256 -> 128 -> 256 -> 784
n_hidden1_units = 256 #隐藏层1神经元数量
n_hidden2_units = 128 #隐藏层2神经元数量

n_input_units = 784 #输入层神经元数量28 * 28
n_output_units = n_input_units #解码器输出层神经元数量，必须等于输入数据的units数量

def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)

def BiasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out]), dtype=tf.float32, name=name_str)

def Enconder(x_origin, activate_func = tf.nn.sigmoid):
    # 编码器第一隐层
    with tf.name_scope('Layer1'):
        weights = WeightsVariable(n_input_units, n_hidden1_units, 'weights')
        biases = BiasesVariable(n_hidden1_units, 'biases')
        x_code1 = activate_func(tf.add(tf.matmul(x_origin, weights), biases))

    with tf.name_scope('Layer2'):
        weights = WeightsVariable(n_hidden1_units, n_hidden2_units, 'weights')
        biases = BiasesVariable(n_hidden2_units, 'biases')
        x_code2 = activate_func(tf.add(tf.matmul(x_code1, weights), biases))
    return x_code2

def Decoder(x_code, activate_func = tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        weights = WeightsVariable(n_hidden2_units, n_hidden1_units, 'weights')
        biases = BiasesVariable(n_hidden1_units, 'biases')
        x_decoder1 = activate_func(tf.add(tf.matmul(x_code, weights), biases))

    with tf.name_scope('Layer2'):
        weights = WeightsVariable(n_hidden1_units, n_output_units, 'weights')
        biases = BiasesVariable(n_output_units, 'biases')
        x_decoder2 = activate_func(tf.add(tf.matmul(x_decoder1, weights), biases))
    return x_decoder2

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.name_scope('X_Origin'):
            X_Origin = tf.placeholder(tf.float32, [None, n_input_units])

        with tf.name_scope('Encoder'):
            X_code = Enconder(X_Origin, activate_func=tf.nn.sigmoid)

        with tf.name_scope('Decoder'):
            X_decode = Decoder(X_code, activate_func=tf.nn.sigmoid)

        with tf.name_scope('Loss'):
            Loss = tf.reduce_mean(tf.pow(X_Origin - X_decode, 2))

        with tf.name_scope('Train'):
            Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            Train = Optimizer.minimize(Loss)

        init = tf.global_variables_initializer()
        print('把计算图写入事件文件，在TensorBoard里面查看')
        writer = tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
        writer.flush()
        mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
        with tf.Session() as sess:
            sess.run(init)
            total_batch = int(mnist.train.num_examples / batch_size)
            for epoch in range(training_epochs):
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, loss = sess.run([Train, Loss], feed_dict={X_Origin : batch_xs})

                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(loss))

            print("模型训练完毕")
            writer.close()

            reconstructions = sess.run(X_decode, feed_dict={X_Origin : mnist.test.images[:examples_to_show]})

            f, a = plt.subplots(2,10, figsize=(10,2))
            for i in range(examples_to_show):
                a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
                a[1][i].imshow(np.reshape(reconstructions[i], (28, 28)))
            f.show()
            plt.draw()
            plt.waitforbuttonpress()