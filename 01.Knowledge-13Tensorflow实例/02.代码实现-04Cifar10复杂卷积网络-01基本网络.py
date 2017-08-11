'''
输入数据->卷积层1->激活层1->池化层1->卷积层2->激活层2->池化层2->
非线性全连接层1->非线性全连接层2->全连接层3->SoftMax->Optimizer
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
import cifar10_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

conv1_kernel_num = 64
conv2_kernel_num = 64
fc1_units_num = 384
fc2_units_num = 192
image_size = 24
image_channel = 3
n_classes = 10

dataset_dir = '../Cifar10_data'
def WeightsVariable(shape, name_str, stddev = 0.1):
    initial = tf.truncated_normal(shape, stddev, dtype=tf.float32)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str, init_value):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name = name_str)

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

def get_distored_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    return images, labels

def get_undistored_eval_batch(eval_data, data_dir, batch_size):
    pass

if __name__ == '__main__':
    with tf.Graph().as_default():
        # 输入
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel],
                                           name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

        #前向推断
        with tf.name_scope('Inference'):
            logits = Inference(images_holder)

        #定义损失层
        with tf.name_scope('Loss'):
            # 因为cifar10不是one-hot编码的，所以不能使用softmax，而sparse内部会进行one-hot编码
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder, logits=logits)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            total_loss = cross_entropy_mean

        #定义优化训练层
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            # optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate, global_step=global_step)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
            '''
            两个步骤：
            1.反向传播计算梯度
            2.利用梯度下降算法优化权重与偏置
            '''
            #minimize更新参数后，这个global_step参数会自动加一,可以当做调用minimize的次数
            trainer = optimizer.minimize(cross_entropy_loss, global_step=global_step)

        #定义模型评估层
        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k = 1)

        with tf.name_scope('GetTrainBatch'):
            images_train, labels_train = get_distored_train_batch(data_dir=dataset_dir, batch_size=batch_size)

        with tf.name_scope('GetTestBatch'):
            images_test, labels_test = get_undistored_eval_batch(eval_data=True, data_dir=dataset_dir,
                                                                 batch_size=batch_size)
        init_op = tf.global_variables_initializer()

        summary_writer = tf.summary.FileWriter(logdir='../logs', graph=tf.get_default_graph())
        summary_writer.close()