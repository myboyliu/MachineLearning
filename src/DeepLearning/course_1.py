import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

learning_rate = 0.02
training_epochs = 3000
display_step=50

train_x = np.asarray([3.3,4.4,5.5,6.71,6.93, 4.168,9.779,6.182,
                      7.59,2.167,7.042,10.791,5.313,7.997,5.645,9.27,3.1])
train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                      2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_x.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b) # 前向计算

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2 * n_samples) #反向计算

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
lineW = 0
lineb = 0
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                    "W=", sess.run(W), "b=", sess.run(b))

    training_cost = sess.run(cost, feed_dict={X : train_x, Y : train_y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    lineW = sess.run(W)
    lineb = sess.run(b)

plt.figure(1, facecolor='white')
plt.scatter(train_x, train_y, c='b')

lineX = np.linspace(3,11, 15)
lineY = (lineW * lineX + lineb)
plt.plot(lineX, lineY, color='b')

plt.show()