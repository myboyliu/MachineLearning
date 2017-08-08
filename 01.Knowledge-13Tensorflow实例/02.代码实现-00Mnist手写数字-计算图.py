import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(_):
    print('~~~~~开始设计计算图')
    with tf.Graph().as_default():
        with tf.name_scope('Input'):
            X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
            Y_true = tf.placeholder(tf.float32, shape=[None, 10], name='Y_true') # 10个神经元

        with tf.name_scope('Inference'):
            W = tf.Variable(tf.zeros([784, 10]), name="Weight")
            b = tf.Variable(tf.zeros([10]), name="Bias")
            logits = tf.add(tf.matmul(X, W), b)
            with tf.name_scope('Softmax'):
                Y_pred = tf.nn.softmax(logits=logits)

        with tf.name_scope('Loss'):
            TrainLoss = tf.reduce_mean(-tf.reduce_sum(Y_true * tf.log(Y_pred), axis=1))

        with tf.name_scope('Train'):
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            TrainOp = Optimizer.minimize(TrainLoss)

        with tf.name_scope('Evaluate'):
            correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        InitOp = tf.global_variables_initializer()
        #
        # with tf.name_scope('Evaluate'):
        #     EvalLoss = tf.reduce_mean(tf.pow((Y_pred - Y_true), 2)) / 2

        print('~~~~~将计算图写入事件文件~~~~~')
        writer = tf.summary.FileWriter(logdir='../logs/mnist_softmax', graph=tf.get_default_graph())
        writer.close()
        print('~~~~~开始运行计算图~~~~~')
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        sess = tf.InteractiveSession()
        sess.run(InitOp)

        for step in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, train_loss, train_W, train_b = sess.run([TrainOp, TrainLoss, W, b],
                                                       feed_dict={X:batch_xs, Y_true: batch_ys})
            # print("train step:", step, ", train_loss:", train_loss)

        accuracy_score = sess.run(accuracy,feed_dict={X: mnist.test.images,
                                                      Y_true: mnist.test.labels})
        y_predd = sess.run(Y_pred,feed_dict={X: mnist.test.images,
                                             W: train_W, b : train_b})
        print("模型准确率：", accuracy_score)
        y_pre = sess.run(tf.argmax(y_predd, 1))
        y_tru = sess.run(tf.argmax(mnist.test.labels, 1))

        result = y_pre == y_tru

        print(len(result[result == True]) / len(y_predd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #为参数解析器添加参数：data_dir(指定数据集存放路径)
    parser.add_argument('--data_dir', type=str,
                        default='../MNIST_data/',  #参数默认值
                        help='数据集存放路径')
    FLAGS, unparsed = parser.parse_known_args() #解析参数
    #运行TensorFlow应用
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)