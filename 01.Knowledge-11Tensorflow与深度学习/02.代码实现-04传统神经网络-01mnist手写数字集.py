'''传统神经网络算法
Iter0, Testing Accurancy 0.8655, Training Accurancy 0.860364, Training Loss:1.63175
Iter1, Testing Accurancy 0.892, Training Accurancy 0.889436, Training Loss:1.58606
Iter2, Testing Accurancy 0.9068, Training Accurancy 0.901836, Training Loss:1.56838
Iter3, Testing Accurancy 0.9118, Training Accurancy 0.909891, Training Loss:1.55852
Iter4, Testing Accurancy 0.9183, Training Accurancy 0.916582, Training Loss:1.55065
Iter5, Testing Accurancy 0.9176, Training Accurancy 0.919673, Training Loss:1.54675
Iter6, Testing Accurancy 0.9213, Training Accurancy 0.923855, Training Loss:1.5413
Iter7, Testing Accurancy 0.9285, Training Accurancy 0.9284, Training Loss:1.537
Iter8, Testing Accurancy 0.9306, Training Accurancy 0.932473, Training Loss:1.5328
Iter9, Testing Accurancy 0.9302, Training Accurancy 0.932855, Training Loss:1.53145
Iter10, Testing Accurancy 0.9312, Training Accurancy 0.933055, Training Loss:1.53086
Iter11, Testing Accurancy 0.9335, Training Accurancy 0.937418, Training Loss:1.52649
Iter12, Testing Accurancy 0.9349, Training Accurancy 0.937673, Training Loss:1.52564
Iter13, Testing Accurancy 0.9396, Training Accurancy 0.9406, Training Loss:1.52336
Iter14, Testing Accurancy 0.9414, Training Accurancy 0.942509, Training Loss:1.52099
Iter15, Testing Accurancy 0.9387, Training Accurancy 0.942364, Training Loss:1.52084
Iter16, Testing Accurancy 0.9414, Training Accurancy 0.943945, Training Loss:1.51931
Iter17, Testing Accurancy 0.9411, Training Accurancy 0.945673, Training Loss:1.51782
Iter18, Testing Accurancy 0.9457, Training Accurancy 0.947655, Training Loss:1.5156
Iter19, Testing Accurancy 0.9461, Training Accurancy 0.947964, Training Loss:1.51513
Iter20, Testing Accurancy 0.9436, Training Accurancy 0.948236, Training Loss:1.51505
Iter21, Testing Accurancy 0.9448, Training Accurancy 0.950164, Training Loss:1.51271
Iter22, Testing Accurancy 0.9498, Training Accurancy 0.950364, Training Loss:1.51268
Iter23, Testing Accurancy 0.9453, Training Accurancy 0.951109, Training Loss:1.51168
Iter24, Testing Accurancy 0.9477, Training Accurancy 0.951964, Training Loss:1.51086
Iter25, Testing Accurancy 0.9488, Training Accurancy 0.9534, Training Loss:1.50916
Iter26, Testing Accurancy 0.9491, Training Accurancy 0.955091, Training Loss:1.50763
Iter27, Testing Accurancy 0.9506, Training Accurancy 0.954545, Training Loss:1.50792
Iter28, Testing Accurancy 0.9493, Training Accurancy 0.955818, Training Loss:1.5069
Iter29, Testing Accurancy 0.9482, Training Accurancy 0.956327, Training Loss:1.5063
Iter30, Testing Accurancy 0.9524, Training Accurancy 0.957436, Training Loss:1.50505
Done
'''
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("../Total_Data/MNIST_data/", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size # //表示整数除法

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10]) # one-hot编码后的输出
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1)) # 不要初始化为0
b1 = tf.Variable(tf.zeros([300]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([300, 150], stddev=0.1)) # 不要初始化为0
b2 = tf.Variable(tf.zeros([150]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([150, 50], stddev=0.1)) # 不要初始化为0
b3 = tf.Variable(tf.zeros([50]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1)) # 不要初始化为0
b4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                              logits=prediction)) #交叉熵明显收敛更快，而且运行速度也快
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x : batch_xs, y : batchys, keep_prob:0.7})
            #print("idx:" + str(batch) + ", Train Loss:" + str(loss) + ", Acc:" + str(acc))

        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,
                                                 y:mnist.test.labels,
                                                 keep_prob:0.7})
        train_acc, train_loss = sess.run([accuracy, loss], feed_dict={x:mnist.train.images,
                                                  y:mnist.train .labels,
                                                  keep_prob:0.7})
        print("Iter" + str(epoch) + ", Testing Accurancy " + str(test_acc) + ", Training Accurancy " + str(train_acc)
              + ", Training Loss:" + str(train_loss))

    print("Done")

