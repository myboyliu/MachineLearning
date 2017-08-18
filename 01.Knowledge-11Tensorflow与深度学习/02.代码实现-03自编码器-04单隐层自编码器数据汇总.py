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

#编码率n_hidden_units / n_input_units
n_hidden_units = 128 #隐藏层神经元数量
n_input_units = 784 #输入层神经元数量28 * 28
n_output_units = n_input_units #解码器输出层神经元数量，必须等于输入数据的units数量

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)

def BiasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out]), dtype=tf.float32, name=name_str)

def Enconder(x_origin, activate_func = tf.nn.sigmoid):
    # 编码器第一隐层
    with tf.name_scope('Layer'):
        weights = WeightsVariable(n_input_units, n_hidden_units, 'weights')
        biases = BiasesVariable(n_hidden_units, 'biases')
        x_code = activate_func(tf.add(tf.matmul(x_origin, weights), biases))
        variable_summaries(weights)
        variable_summaries(biases)
    return x_code

def Decoder(x_code, activate_func = tf.nn.sigmoid):
    with tf.name_scope('Layer'):
        weights = WeightsVariable(n_hidden_units, n_output_units, 'weights')
        biases = BiasesVariable(n_output_units, 'biases')
        x_decoder = activate_func(tf.add(tf.matmul(x_code, weights), biases))
        variable_summaries(weights)
        variable_summaries(biases)
    return x_decoder

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.name_scope('X_Origin'):
            X_Origin = tf.placeholder(tf.float32, [None, n_input_units])

        with tf.name_scope('Encoder'):
            X_code = Enconder(X_Origin)

        with tf.name_scope('Decoder'):
            X_decode = Decoder(X_code)

        with tf.name_scope('Loss'):
            Loss = tf.reduce_mean(tf.pow(X_Origin - X_decode, 2))

        with tf.name_scope('Train'):
            Optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            Train = Optimizer.minimize(Loss)

        with tf.name_scope("LossSummary"):
            tf.summary.scalar('loss', Loss)
            tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope("image_summaries"):
            image_original = tf.reshape(X_Origin, [-1,28,28,1])
            image_reconstructed = tf.reshape(X_decode, [-1,28,28,1])
            tf.summary.image('image_original', image_original, 10)
            tf.summary.image('image_reconstructed', image_reconstructed, 10)

        merged_summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        print('把计算图写入事件文件，在TensorBoard里面查看')
        writer = tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
        writer.flush()
        mnist = input_data.read_data_sets('../Total_Data/MNIST_data/', one_hot=True)
        with tf.Session() as sess:
            sess.run(init)
            total_batch = int(mnist.train.num_examples / batch_size)
            for epoch in range(training_epochs):
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, loss = sess.run([Train, Loss], feed_dict={X_Origin : batch_xs})

                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(loss))
                    summary_str = sess.run(merged_summary, feed_dict={X_Origin : batch_xs})
                    writer.add_summary(summary_str, epoch)
                    writer.flush()

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