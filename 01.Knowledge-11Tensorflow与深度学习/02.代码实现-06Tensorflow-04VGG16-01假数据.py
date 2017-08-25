'''
VGG模型
输入层224*224*3

卷积层1：3*3，64，1 输出尺寸：224*224*64
卷积层2：3*3，64，1 输出尺寸：224*224*64
池化层1：2*2，maxpool，2 输出尺寸：112*112*64

卷积层3：3*3，128，1
卷积层4：3*3，128，1
池化层2：2*2，maxpool，2 输出尺寸：56*56*128

卷积层5：3*3，256，1
卷积层6：3*3，256，1
卷积层7：3*3，256，1
池化层3：2*2，maxpool，2 输出尺寸28*28*256

卷积层8：3*3，512，1
卷积层9：3*3，512，1
卷积层10：3*3，512，1
池化层4：2*2，maxpool，2 输出尺寸14*14*512

卷积层11：3*3，512，1
卷积层12：3*3，512，1
卷积层13：3*3，512，1
池化层5：2*2，maxpool，2 输出尺寸7*7*512 = 25088

非线性全连接层1：4096
Dropout层1：训练时0.5，预测时1.0
非线性全连接层2：4096
Dropout层2：训练时0.5，预测时1.0
线性全连接层1：1000
softmax
'''

import tensorflow as tf
import numpy as np

N_CLASSES = 1000
IMAGE_SIZE = 224
IMAGE_DEEPTH = 3

LEARNING_RATE_INIT = 0.001
BATCH_SIZE = 32
DISPLAY_STEP = 10

LAYER1_NUMBER = 64
LAYER2_NUMBER = 128
LAYER3_NUMBER = 256
LAYER4_NUMBER = 512
LAYER5_NUMBER = 512
LAYER6_NUMBER = 4096

TRAINING_EPOCHS = 10
NUM_BATCHES_PER_EPOCH_FOR_TRAIN = 1000
NUM_BATCHES_PER_EPOCH_FOR_EVAL = 500

def getFakeData():
    images = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEEPTH], mean=0.0, stddev=1.0, dtype=tf.float32))
    labels = tf.Variable(tf.random_uniform(shape=[BATCH_SIZE], minval=0, maxval=N_CLASSES,dtype=tf.int32))
    return images, labels

with tf.name_scope('Inputs'):
    images_holder = tf.placeholder(shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEEPTH], dtype=tf.float32, name='images')
    labels_holder = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32, name='labels')

with tf.name_scope('Inference'):
    with tf.name_scope('Layer1'):
        with tf.name_scope('Conv2d_1'):
            initial = tf.truncated_normal(shape=[3, 3, IMAGE_DEEPTH, LAYER1_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.0, shape=[LAYER1_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            conv1_out = tf.nn.conv2d(images_holder, weights, strides=[1, 1,1,1], padding='SAME')
            conv1_out = tf.nn.bias_add(conv1_out, biases)
            conv1_out = tf.nn.relu(conv1_out)
            print(conv1_out.op.name, ' ', conv1_out.get_shape().as_list())
        with tf.name_scope('Conv2d_2'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER1_NUMBER, LAYER1_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER1_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv2_out = tf.nn.conv2d(conv1_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv2_out = tf.nn.bias_add(conv2_out, biases)
            conv2_out = tf.nn.relu(conv2_out)
            print(conv2_out.op.name, ' ', conv2_out.get_shape().as_list())
        with tf.name_scope('Pool_1'):
            pool1_out = tf.nn.max_pool(conv2_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Layer2'):
        with tf.name_scope('Conv2d_3'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER1_NUMBER, LAYER2_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.0, shape=[LAYER2_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            conv3_out = tf.nn.conv2d(pool1_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv3_out = tf.nn.bias_add(conv3_out, biases)
            conv3_out = tf.nn.relu(conv3_out)
            print(conv3_out.op.name, ' ', conv3_out.get_shape().as_list())
        with tf.name_scope('Conv2d_4'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER2_NUMBER, LAYER2_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER2_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv4_out = tf.nn.conv2d(conv3_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv4_out = tf.nn.bias_add(conv4_out, biases)
            conv4_out = tf.nn.relu(conv4_out)
            print(conv4_out.op.name, ' ', conv4_out.get_shape().as_list())
        with tf.name_scope('Pool_2'):
            pool2_out = tf.nn.max_pool(conv4_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Layer3'):
        with tf.name_scope('Conv2d_5'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER2_NUMBER, LAYER3_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.0, shape=[LAYER3_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            conv5_out = tf.nn.conv2d(pool2_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv5_out = tf.nn.bias_add(conv5_out, biases)
            conv5_out = tf.nn.relu(conv5_out)
            print(conv5_out.op.name, ' ', conv5_out.get_shape().as_list())
        with tf.name_scope('Conv2d_6'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER3_NUMBER, LAYER3_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER3_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv6_out = tf.nn.conv2d(conv5_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv6_out = tf.nn.bias_add(conv6_out, biases)
            conv6_out = tf.nn.relu(conv6_out)
            print(conv6_out.op.name, ' ', conv6_out.get_shape().as_list())
        with tf.name_scope('Conv2d_7'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER3_NUMBER, LAYER3_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER3_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv7_out = tf.nn.conv2d(conv6_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv7_out = tf.nn.bias_add(conv7_out, biases)
            conv7_out = tf.nn.relu(conv7_out)
            print(conv7_out.op.name, ' ', conv7_out.get_shape().as_list())
        with tf.name_scope('Pool_3'):
            pool3_out = tf.nn.max_pool(conv7_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Layer4'):
        with tf.name_scope('Conv2d_8'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER3_NUMBER, LAYER4_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.0, shape=[LAYER4_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            conv8_out = tf.nn.conv2d(pool3_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv8_out = tf.nn.bias_add(conv8_out, biases)
            conv8_out = tf.nn.relu(conv8_out)
            print(conv8_out.op.name, ' ', conv8_out.get_shape().as_list())
        with tf.name_scope('Conv2d_9'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER4_NUMBER, LAYER4_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER4_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv9_out = tf.nn.conv2d(conv8_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv9_out = tf.nn.bias_add(conv9_out, biases)
            conv9_out = tf.nn.relu(conv9_out)
            print(conv9_out.op.name, ' ', conv9_out.get_shape().as_list())
        with tf.name_scope('Conv2d_10'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER4_NUMBER, LAYER4_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER4_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv10_out = tf.nn.conv2d(conv9_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv10_out = tf.nn.bias_add(conv10_out, biases)
            conv10_out = tf.nn.relu(conv10_out)
            print(conv10_out.op.name, ' ', conv10_out.get_shape().as_list())
        with tf.name_scope('Pool_3'):
            pool4_out = tf.nn.max_pool(conv10_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Layer5'):
        with tf.name_scope('Conv2d_11'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER4_NUMBER, LAYER5_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.0, shape=[LAYER5_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            conv11_out = tf.nn.conv2d(pool4_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv11_out = tf.nn.bias_add(conv11_out, biases)
            conv11_out = tf.nn.relu(conv11_out)
            print(conv11_out.op.name, ' ', conv11_out.get_shape().as_list())
        with tf.name_scope('Conv2d_12'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER5_NUMBER, LAYER5_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER5_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv12_out = tf.nn.conv2d(conv11_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv12_out = tf.nn.bias_add(conv12_out, biases)
            conv12_out = tf.nn.relu(conv12_out)
            print(conv12_out.op.name, ' ', conv12_out.get_shape().as_list())
        with tf.name_scope('Conv2d_13'):
            initial = tf.truncated_normal(shape=[3, 3, LAYER5_NUMBER, LAYER5_NUMBER], stddev=1e-1,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights2')
            initial = tf.constant(0.0, shape=[LAYER5_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases2')
            conv13_out = tf.nn.conv2d(conv12_out, weights, strides=[1, 1,1,1], padding='SAME')
            conv13_out = tf.nn.bias_add(conv13_out, biases)
            conv13_out = tf.nn.relu(conv13_out)
            print(conv13_out.op.name, ' ', conv13_out.get_shape().as_list())
        with tf.name_scope('Pool_3'):
            pool5_out = tf.nn.max_pool(conv13_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Features'):
        features = tf.reshape(pool5_out, shape=[BATCH_SIZE, -1])
        feats_dim = features.get_shape()[1].value

    with tf.name_scope('Layer6'):
        with tf.name_scope('FC_Nonlinear1'):
            initial = tf.truncated_normal(shape=[feats_dim, LAYER6_NUMBER], stddev=0.04,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.1, shape=[LAYER6_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            fc1_out = tf.matmul(features, weights)
            fc1_out = tf.add(fc1_out, biases)
            fc1_out = tf.nn.relu(fc1_out)
            print(fc1_out.op.name, ' ', fc1_out.get_shape().as_list())
        with tf.name_scope('Dropout1'):
            keep_prob = tf.placeholder(dtype=tf.float32)
            dropout1 = tf.nn.dropout(fc1_out, keep_prob=keep_prob)
        with tf.name_scope('FC_Nonlinear2'):
            initial = tf.truncated_normal(shape=[LAYER6_NUMBER,LAYER6_NUMBER], stddev=0.04,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.1, shape=[LAYER6_NUMBER])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            fc2_out = tf.matmul(dropout1, weights)
            fc2_out = tf.add(fc2_out, biases)
            fc2_out = tf.nn.relu(fc2_out)
            print(fc2_out.op.name, ' ', fc2_out.get_shape().as_list())
        with tf.name_scope('Dropout1'):
            dropout2 = tf.nn.dropout(fc2_out, keep_prob=keep_prob)
        with tf.name_scope('FC_Linear'):
            initial = tf.truncated_normal(shape=[LAYER6_NUMBER,N_CLASSES], stddev=0.04,dtype=tf.float32)
            weights = tf.Variable(initial_value=initial, dtype=tf.float32, name='weights1')
            initial = tf.constant(0.1, shape=[N_CLASSES])
            biases = tf.Variable(initial_value=initial, dtype=tf.float32, name='biases1')
            fc3_out = tf.matmul(dropout2, weights)
            fc3_out = tf.add(fc3_out, biases)
            logits = tf.identity(fc3_out)
            print(logits.op.name, ' ', logits.get_shape().as_list())

with tf.name_scope('Loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_holder)
    total_loss = tf.reduce_mean(cross_entropy)

with tf.name_scope('Train'):
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE_INIT).minimize(total_loss, global_step=global_step)

with tf.name_scope('Evaluate'):
    top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

with tf.name_scope('GetTrainBatch'):
    images_train, labels_train = getFakeData()

with tf.name_scope('GetTestBatch'):
    images_test, labels_test = getFakeData()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_batches_per_epoch = int(NUM_BATCHES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE)
    for epoch in range(TRAINING_EPOCHS):
        for batch_idx in range(num_batches_per_epoch):
            print(batch_idx)
            images_batch, labels_batch = sess.run([images_train, labels_train])
            print(labels_batch)
            _, loss_value = sess.run([train_op, total_loss],
                                     feed_dict={images_holder: images_batch,
                                                labels_holder:labels_batch,
                                                keep_prob:0.5})
            training_step = sess.run(global_step)
            if training_step % DISPLAY_STEP == 0 :
                predictions = sess.run([top_K_op], feed_dict={images_holder:images_batch,
                                                              labels_holder:labels_batch,
                                                              keep_prob:0.5})
                batch_accuracy = np.sum(predictions) * 1.0 / BATCH_SIZE

                print("Training Epoch: " + str(epoch) +
                      ", Training Step: " + str(training_step) +
                      ", Training Loss= " + "{:.6f}".format(loss_value) +
                      ", Training Accuracy= " + "{:.5f}".format(batch_accuracy))

    print('训练完毕')