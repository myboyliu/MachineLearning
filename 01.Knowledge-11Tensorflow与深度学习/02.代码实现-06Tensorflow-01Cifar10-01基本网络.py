'''
输入数据->卷积层1->激活层1->池化层1->卷积层2->激活层2->池化层2->非线性全连接层1->非线性全连接层2->全连接层3->SoftMax->Optimizer
输入数据: 24 * 24 * 3 (cifar10的图片都是32*32*3的，需要处理成24*24*3)
卷积层1：5*5 卷积核个数为K1 步长为1，输出为24 * 24 * K1
激活层1：ReLU
池化层1：3*3 步长为2，输出为12 * 12 * K1
卷积层2：5*5 卷积核个数为K2 步长为1，12 * 12 * K2
激活层2：ReLU
池化层2：3*3 步长为2 输出为6 * 6 * K2
非线性全连接层1：神经元个数200(这一层相当于有200*6*6*K2个权重，以及200个偏置)，输出为200
非线性全连接层1：神经元个数100(这一层相当于有100*200个权重，以及100个偏置)，输出为100
线性全连接层：神经元个数10
softmax层
'''
import tensorflow as tf
import os
import cifar_input
import numpy as np
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

dataset_dir = '../Total_Data/Cifar10_data'
num_examples_per_epoch_for_train = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN # 50000
num_examples_per_epoch_for_eval = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
image_size = cifar_input.IMAGE_SIZE
image_channel = 3
n_classes = cifar_input.NUM_CLASSES_CIFAR10

conv1_kernel_num = 32
conv2_kernel_num = 32
fc1_units_num = 192
fc2_units_num = 96

def WeightsVariable(shape, name_str, stddev = 0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str, init_value):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name = name_str)

# 卷积层不做降采样
def Conv2d(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y

def Pool2d(x, pool = tf.nn.max_pool, k =2, stride=2, padding='SAME'):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)

def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y

def Inference(images_holder):
    with tf.name_scope('Conv2d_1'): # 卷积层1
        weights = WeightsVariable(shape=[5, 5, image_channel, conv1_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernel_num], name_str='biases', init_value=0.0)
        conv1_out = Conv2d(images_holder, weights, biases, stride=1, padding='SAME')

    with tf.name_scope('Pool2d_1'): #池化层1
        pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

    with tf.name_scope('Conv2d_2'): # 卷积层2
        weights = WeightsVariable(shape=[5, 5, conv1_kernel_num, conv2_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv2_kernel_num], name_str='biases', init_value=0.0)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')

    with tf.name_scope('Pool2d_2'): #池化层2
        pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME') #6 * 6 * 64

    with tf.name_scope('FeatsReshape'): #将二维特征图变为一维特征向量，得到的是conv1_kernel_num个特征图，每个特征图是12*12的
        features = tf.reshape(pool2_out, [batch_size, -1]) # [batch_size, 2304] 2304 = 6 * 6 * 64
        feats_dim = features.get_shape()[1].value

    with tf.name_scope('FC1_nonlinear'): #非线性全连接层1
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num], name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)
        fc1_out = FullyConnected(features, weights, biases,
                                      activate=tf.nn.relu, act_name='relu')

    with tf.name_scope('FC2_nonlinear'): #非线性全连接层2
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num], name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)
        fc2_out = FullyConnected(fc1_out, weights, biases,
                                 activate=tf.nn.relu, act_name='relu')

    with tf.name_scope('FC2_linear'): #线性全连接层
        weights = WeightsVariable(shape=[fc2_units_num, n_classes], name_str='weights', stddev=1.0 / fc2_units_num)
        biases = BiasesVariable(shape=[n_classes], name_str='biases', init_value=0.0)
        logits = FullyConnected(fc2_out, weights, biases,
                                 activate=tf.identity, act_name='linear')

    return logits

'''
返回的images是[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
返回的labels不是one-hot编码的，因为它返回的是[batch_size]，而不是[batch_size, n_classes]
'''
def get_distored_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar_input.distorted_inputs(cifar10or20or100=10, data_dir=data_dir, batch_size=batch_size)
    return images, labels

'''
获取评估测试集
'''
def get_undistored_eval_batch(eval_data, data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar_input.inputs(cifar10or20or100=10, eval_data=eval_data, data_dir=data_dir, batch_size=batch_size)
    return images, labels

if __name__ == '__main__':
    with tf.Graph().as_default():
        # 输入
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel],
                                           name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')# 0 ~ 9的数字

        #前向推断
        with tf.name_scope('Inference'):
            logits = Inference(images_holder)

        #定义损失层
        with tf.name_scope('Loss'):
            # 因为cifar10不是one-hot编码的，所以不能使用softmax，而sparse内部会进行one-hot编码
            labels = tf.cast( labels_holder, tf.int64 )
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            total_loss = cross_entropy_mean

        #定义优化训练层
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        #定义模型评估层
        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k = 1)

        with tf.name_scope('GetTrainBatch'):
            images_train, labels_train = get_distored_train_batch(data_dir=dataset_dir, batch_size=batch_size)

        with tf.name_scope('GetTestBatch'):
            images_test, labels_test = get_undistored_eval_batch(eval_data=True, data_dir=dataset_dir,
                                                                 batch_size=batch_size)
        init_op = tf.global_variables_initializer()

        # summary_writer = tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
        # summary_writer.close()

        results_list = list()
        results_list.append(['learning_rate', learning_rate_init,
                             'training_epochs', training_epochs,
                             'batch_size', batch_size,
                             'display_step', display_step,
                             'conv1_kernel_num', conv1_kernel_num,
                             'conv2_kernel_num', conv2_kernel_num,
                             'fc1_units_num', fc1_units_num,
                             'fc2_units_num', fc2_units_num])
        results_list.append(['train_step', 'train_loss', 'train_step', 'train_accuracy'])

        with tf.Session() as sess:
            sess.run(init_op)
            print('==>>>>>>>>>>==开始在训练集上训练模型==<<<<<<<<<<==')
            total_batches = int(num_examples_per_epoch_for_train / batch_size)
            print('Per batch Size: ', batch_size)
            print('Train sample Count Per Epoch: ', num_examples_per_epoch_for_train)
            print('Total batch Count Per Epoch: ', total_batches)

            tf.train.start_queue_runners()
            training_step = 0
            for epoch in range(training_epochs):
                for batch_idx in range(total_batches):
                    images_batch, label_batch = sess.run([images_train, labels_train])
                    # print(label_batch)
                    _, loss_value = sess.run([train_op, total_loss], feed_dict={images_holder: images_batch,
                                                                                labels_holder: label_batch,
                                                                                learning_rate:learning_rate_init})
                    training_step = sess.run(global_step)
                    if training_step % display_step == 0:

                        predictions = sess.run([top_K_op], feed_dict={images_holder: images_batch,
                                                                      labels_holder : label_batch})
                        batch_accuracy = np.sum(predictions) / batch_size
                        results_list.append([training_step, loss_value, training_step, batch_accuracy])
                        print("Training Step: " + str(training_step) +
                              ", Training Loss= " + "{:.6f}".format(loss_value) +
                              ", Training Accuracy= " + "{:.5f}".format(batch_accuracy))
            print('训练完毕！')

            print('==>>>>>>>>>>==开始在测试集上评估模型==<<<<<<<<<<==')
            total_batches = int(num_examples_per_epoch_for_eval / batch_size)
            total_examples = total_batches * batch_size
            print('Per batch Size: ', batch_size)
            print('Test sample Count Per Epoch: ', total_examples)
            print('Total batch Count Per Epoch: ', total_batches)

            correct_predicted = 0
            for test_step in range(total_batches):
                images_batch, label_batch = sess.run([images_test, labels_test])
                predictions = sess.run([top_K_op], feed_dict={images_holder: images_batch,
                                                              labels_holder: label_batch})
                correct_predicted += np.sum(predictions)
            accuracy_score = correct_predicted / total_examples
            print('--------->Accuracy on Test Examples: ', accuracy_score)
            results_list.append(['Accuracy on Test Examples: ', accuracy_score])

            results_file = open('../logs/SummaryFiles/result_0111020601.csv', 'w', newline='')
            csv_writer = csv.writer(results_file, dialect='excel')
            for row in results_list:
                csv_writer.writerow(row)
