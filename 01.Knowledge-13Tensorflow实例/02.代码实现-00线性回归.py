import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib as mpl

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

N = 10
N1 = 6
x = np.linspace(-3, 3, N)
rng = np.random.RandomState(42)
y = np.sin(4 * x) + x + rng.uniform(size=len(x))

X = x[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)
print('~~~~~~~开始设计计算图~~~~~~~')
with tf.Graph().as_default():
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32)
        Y_true = tf.placeholder(tf.float32)

    with tf.name_scope('Inference'):
        W = tf.Variable(tf.zeros([1]))
        b = tf.Variable(tf.zeros([1]))

        # 前向预测模型 inference
        Y_pred = tf.add(tf.multiply(X, W), b) # w * x + b

    with tf.name_scope('Loss'):
        # Loss - MSE损失
        TrainLoss = tf.reduce_mean(tf.pow(Y_true - Y_pred, 2)) / 2

    with tf.name_scope('Train'):
        # 反向梯度计算
        Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        TrainOp = Optimizer.minimize(TrainLoss)

    with tf.name_scope('Evaluate'):
        # 添加评估节点
        EvaluteLoss = tf.reduce_mean(tf.pow(Y_true - Y_pred, 2)) / 2

    InitOp = tf.global_variables_initializer()

    # writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    # writer.close()
    print('~~~~~~~设计计算图结束~~~~~~~')
    print('~~~~~~~开启会话~~~~~~~')
    sess = tf.Session()
    sess.run(InitOp)

    for step in range(1000):
        for tx, ty in zip(X_train, y_train):
            _, train_loss, train_w, train_b = sess.run([TrainOp, TrainLoss, W, b], feed_dict={X:tx, Y_true:ty})

        if (step + 1) % 100 == 0:
            print("Step:", "%04d" % (step + 1), "train_loss=", "{:.9f}".format(train_loss), "W=", train_w, ", b=", train_b)

    print('~~~~~~~训练结束~~~~~~~')
    plt.plot(X_train, y_train, 'o')
    plt.plot(X_train, train_w * X_train + train_b, 'g-', linewidth=2, label="1阶")

    plt.legend(loc='best')
    plt.show()