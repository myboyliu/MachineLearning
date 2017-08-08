import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/iris.data', header=None)
x = data.values[:, :2]
y = pd.Categorical(data[4]).codes

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=123,
                                                    stratify=y)
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

print('Xtrain.shape:', train_x.shape, ', Xtest.shape:', test_x.shape)
print('Ytrain.shape:', train_y.shape, ', Ytest.shape:', test_y.shape)
#
xtrain = tf.placeholder("float", [None, train_x.shape[1]])
xtest = tf.placeholder("float", [test_x.shape[1]])
#
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)
pred = tf.arg_min(distance, 0)
#
accuracy = 0

init = tf.global_variables_initializer()

error_list = []
with tf.Session() as sess:
    sess.run(init)
    Ntest = len(test_x)
    for i in range(Ntest):
        nn_index = sess.run(pred, feed_dict={xtrain:train_x, xtest:test_x[i, :]})
        pred_class_label = np.argmax(train_y[nn_index])
        true_class_label = np.argmax(test_y[i])

        if pred_class_label == true_class_label:
            accuracy += 1
        else:
            error_list.append([pred_class_label, true_class_label, i])
    print("Done!")
    accuracy /= Ntest
    print("Accuracy: %.2f%%" % (100 * accuracy))
    print(error_list)