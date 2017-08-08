from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print('Xtrain.shape:', Xtrain.shape, ', Xtest.shape:', Xtest.shape)
print('Ytrain.shape:', Ytrain.shape, ', Ytest.shape:', Ytest.shape)

xtrain = tf.placeholder("float", [None, 784])
xtest = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)
pred = tf.arg_min(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()

error_list = []
with tf.Session() as sess:
    sess.run(init)
    Ntest = len(Xtest)
    for i in range(Ntest):
        nn_index = sess.run(pred, feed_dict={xtrain:Xtrain, xtest:Xtest[i, :]})
        pred_class_label = np.argmax(Ytrain[nn_index])
        true_class_label = np.argmax(Ytest[i])

        if pred_class_label == true_class_label:
            accuracy += 1
        else:
            error_list.append([pred_class_label, true_class_label, i])
    print("Done!")
    accuracy /= Ntest
    print("Accuracy: %.2f%%" % (100 * accuracy))
    print(error_list)
    plt.figure(figsize=(14, 5), facecolor='w')

    h_count = 6
    v_count = int(len(error_list) / h_count + 1)
    print(v_count)
    errIndex = [x[2] for x in error_list]
    # errIndex=[1,2,3,4,5,6,7,8]
    print(errIndex)
    index1 = 0
    for index, image in enumerate(mnist.test.images):
        if index in errIndex:
            plt.subplot(v_count, h_count, index1 + 1)
            first_image = np.array(image)
            pixels = first_image.reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.title(u'错分为：%i，真实值：%i' % (error_list[index1][0], error_list[index1][1]))
            index1 += 1

        if index1 >= len(errIndex):
            break
    plt.tight_layout()
    plt.show()