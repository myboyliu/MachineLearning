'''
cifar10的数据，cifar20的数据，以及cifar100的数据，cifar图片是32*32*3的，所以一些参数需要修改
'''
import numpy as np
import tensorflow as tf
import os
import cifar_input
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cifar10or20or100 = 10
num_examples_per_epoch_for_train = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
dataset_dir_cifar10 = '../Total_Data/TempData/cifar-10-batches-bin'
dataset_dir_cifar100= '../Total_Data/TempData/cifar-100-binary-bin'

learning_rate_init = 0.001
training_epochs = 5
batch_size = 100
display_step = 1
conv1_kernel_num = 96
conv2_kernel_num = 256
conv3_kernel_num = 384
conv4_kernel_num = 384
conv5_kernel_num = 256
fc1_units_num = 4096
fc2_units_num = 4096

image_size = cifar_input.IMAGE_SIZE
image_channel = cifar_input.IMAGE_DEPTH

if cifar10or20or100 == 10:
    n_classes = cifar_input.NUM_CLASSES_CIFAR10
    dataset_dir = dataset_dir_cifar10
    cifar_input.maybe_download_and_extract('../Total_Data/TempData', cifar_input.CIFAR10_DATA_URL)
elif cifar10or20or100 == 20:
    n_classes = cifar_input.NUM_CLASSES_CIFAR20
    dataset_dir = dataset_dir_cifar100
    cifar_input.maybe_download_and_extract('../Total_Data/TempData', cifar_input.CIFAR100_DATA_URL)
else:
    n_classes = cifar_input.NUM_CLASSES_CIFAR100
    dataset_dir = dataset_dir_cifar100
    cifar_input.maybe_download_and_extract('../Total_Data/TempData', cifar_input.CIFAR100_DATA_URL)

def WeightsVariable(shape, name_str, stddev=0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str, init_value = 0.0):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def Conv2d(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
    with tf.name_scope(act_name):
        y = activation(y)
    return y

def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return pool(x, ksize=[1,k,k,1], strides=[1,stride,stride,1], padding=padding)

def FullyConnected(x, W, b, activate = tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)

    return y

def Inference(images_holder):
    with tf.name_scope('Conv2d_1'): # 5 * 5, 96个，步长1，激活ReLU
        weights = WeightsVariable(shape=[5,5,image_channel, conv1_kernel_num], name_str='weights',stddev=1e-1)
        biases = BiasesVariable(shape=[conv1_kernel_num], name_str='biases', init_value=0.0)
        conv1_out = Conv2d(images_holder, weights, biases, stride=1, padding='SAME')
        print_activations(conv1_out)

    with tf.name_scope('LRNormal_1'): # LRN层，但是其实LRN层并没有太大的作用
        normal1_out = tf.nn.lrn(conv1_out, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        print_activations(normal1_out)

    with tf.name_scope('Pool2d_1'):# 3 * 3, 步长2 重叠池化
        pool1_out = Pool2d(normal1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')
        print_activations(pool1_out)

    with tf.name_scope('Conv2d_2'): # 5 * 5, 256个， 步长1
        weights = WeightsVariable(shape=[5,5,conv1_kernel_num, conv2_kernel_num], name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv2_kernel_num], name_str='biases', init_value=0.0)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')
        print_activations(conv2_out)

    with tf.name_scope('LRNormal_2'): # LRN层，但是其实LRN层并没有太大的作用
        normal2_out = tf.nn.lrn(conv2_out, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        print_activations(normal2_out)

    with tf.name_scope('Pool2d_2'):# 3 * 3, 步长2
        pool2_out = Pool2d(normal2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')
        print_activations(pool2_out)

    with tf.name_scope('Conv2d_3'): # 3 * 3, 384个， 步长1
        weights = WeightsVariable(shape=[3,3,conv2_kernel_num, conv3_kernel_num], name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv3_kernel_num], name_str='biases', init_value=0.0)
        conv3_out = Conv2d(pool2_out, weights, biases, stride=1, padding='SAME')
        print_activations(conv3_out)

    with tf.name_scope('Conv2d_4'): # 3 * 3, 384个， 步长1
        weights = WeightsVariable(shape=[3,3,conv3_kernel_num, conv4_kernel_num], name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv4_kernel_num], name_str='biases', init_value=0.0)
        conv4_out = Conv2d(conv3_out, weights, biases, stride=1, padding='SAME')
        print_activations(conv4_out)

    with tf.name_scope('Conv2d_5'): # 3 * 3, 256个， 步长1
        weights = WeightsVariable(shape=[3,3,conv4_kernel_num, conv5_kernel_num], name_str='weights', stddev=1e-1)
        biases = BiasesVariable(shape=[conv5_kernel_num], name_str='biases', init_value=0.0)
        conv5_out = Conv2d(conv4_out, weights, biases, stride=1, padding='SAME')
        print_activations(conv5_out)

    with tf.name_scope('Pool2d_3'):# 3 * 3, 步长2
        pool3_out = Pool2d(conv5_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')
        print_activations(pool3_out)

    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool3_out, [batch_size, -1])
        feats_dim = features.get_shape()[1].value # 8*8*conv2_kernel_num
        print_activations(features)

    with tf.name_scope('FC_nonlinear1'):
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num], name_str='weights', stddev=0.04)
        biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)
        fc1_out = FullyConnected(features, weights, biases, activate=tf.nn.relu, act_name='relu')
        print_activations(fc1_out)

    with tf.name_scope('FC_nonlinear2'):
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num], name_str='weights', stddev=0.04)
        biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)
        fc2_out = FullyConnected(fc1_out, weights, biases, activate=tf.nn.relu, act_name='relu')
        print_activations(fc2_out)

    with tf.name_scope('FC_linear'):
        weights = WeightsVariable(shape=[fc2_units_num, n_classes],
                                  name_str='weights', stddev=1.0/fc2_units_num)
        biases = BiasesVariable(shape=[n_classes], name_str='biases', init_value=0.0)
        logits = FullyConnected(fc2_out, weights, biases, activate=tf.identity, act_name='linear')
        print_activations(logits)
    return logits

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def get_distored_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = cifar_input.distorted_inputs(cifar10or20or100=cifar10or20or100,
                                                  data_dir=data_dir,
                                                  batch_size=batch_size)
    return images, labels #labels没有进行one-hot编码

def get_undistorted_eval_batch(data_dir, eval_data, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    images, labels = cifar_input.inputs(cifar10or20or100=cifar10or20or100,
                                        eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
    return images, labels#labels没有进行one-hot编码

with tf.Graph().as_default():
    with tf.name_scope('Inputs'):
        images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel], name='images')
        #没有one-hot编码，如果经过了one-hot，那么shape就应该是[batch_size, n_classes]
        labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

    with tf.name_scope('Inference'):
        logits = Inference(images_holder)

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder,
                                                                       logits=logits)
        total_loss_op = tf.reduce_mean(cross_entropy)

    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
        train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss_op,
                                                                                   global_step=global_step)
    with tf.name_scope('Evaluate'):
        top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

    with tf.name_scope('GetTrainBatch'):
        images_train, labels_train = get_distored_train_batch(data_dir=dataset_dir,batch_size=batch_size)
        tf.summary.image('images', images_train, max_outputs=9)

    with tf.name_scope('GetTestBatch'):
        images_test, labels_test = get_undistorted_eval_batch(data_dir=dataset_dir, eval_data=True, batch_size=batch_size)
        tf.summary.image('images', images_test, max_outputs=9)

    init_op = tf.global_variables_initializer()
    results_list = list()
    results_list.append(['TrainStep','TrainLoss','TrainStep','TrainAccuracy'])

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.start_queue_runners()
        print('==>>>>>>>>>>开始在训练集上训练模型<<<<<<<<<<==')
        num_batches_per_epoch = int(num_examples_per_epoch_for_train / batch_size) # 500
        print('Per batch Size: ', batch_size)
        print('Train sample Count Per Epoch: ', num_examples_per_epoch_for_train)
        print('Total batch Count Per Epoch: ', num_batches_per_epoch)
        for epoch in range(training_epochs):
            for batch_idx in range(num_batches_per_epoch):
                images_batch, labels_batch = sess.run([images_train, labels_train])
                _, loss_value = sess.run([train_op, total_loss_op],
                                                   feed_dict={images_holder: images_batch,
                                                              labels_holder:labels_batch,
                                                              learning_rate:learning_rate_init})
                training_step = sess.run(global_step)
                if training_step % display_step == 0 :
                    predictions = sess.run([top_K_op], feed_dict={images_holder:images_batch,
                                                                  labels_holder:labels_batch})
                    batch_accuracy = np.sum(predictions) * 1.0 / batch_size

                    print("Training Epoch: " + str(epoch) +
                          ", Training Step: " + str(training_step) +
                          ", Training Loss= " + "{:.6f}".format(loss_value) +
                          ", Training Accuracy= " + "{:.5f}".format(batch_accuracy))
                    results_list.append([training_step, loss_value, training_step, batch_accuracy])

        print('训练完毕')
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
        # results_list.append(['Accuracy on Test Examples: ', accuracy_score])
        # results_file = open('01.csv', 'w', newline='')
        # csv_writer = csv.writer(results_file, dialect='excel')
        # for row in results_list:
        #     csv_writer.writerow(row)



