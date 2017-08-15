import os
import sys
import tarfile
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_DEPTH = 3
NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR20 = 20
NUM_CLASSES_CIFAR100 = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 #训练集
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 #测试集，评估集

CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/ciar-10-binary.tar.gz'
CIFAR100_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar_100-binary.tar.gz'
print('调用我啦...cifar_input...')

def maybe_download_and_extract(data_dir, data_url = CIFAR10_DATA_URL):
    dest_directory = data_dir
    DATA_URL = data_url
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_cifar10(filename_queue, coarse_or_fine=None):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = IMAGE_DEPTH

    '''
    数据格式
    label是1个字节，数据是3072个字节=32*32*3
    <1 x label><3072 x pixel>
    ...
    <1 x label><3072 x pixel>
    '''
    label_bytes = 1 #类别标签字节数
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    #创建一个固定长度的读取器，读取一个样本记录的所有字节
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8) # 无符号8位，正好是一个字节
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width]) # 这种顺序，是Caffee的顺序
    # 将图像的空间位置和深度位置由[depth, height, width]转换成[height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result

# coarse_or_fine:区别粗细分类，也就是区分是按照20读取，还是100类读取
def read_cifar100(filename_queue, coarse_or_fine='fine'):
    class CIFAR100Record(object):
        pass
    result = CIFAR100Record()
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = IMAGE_DEPTH

    '''
    数据格式
    第一个字节是粗略分类标签
    第二个字节是精细分类标签
    剩下的是图片像素
    '''
    coarse_label_bytes = 1
    fine_label_bytes = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes

    #创建一个固定长度的读取器，读取一个样本记录的所有字节
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8) # 无符号8位，正好是一个字节

    coarse_label = tf.cast(tf.strided_slice(record_bytes, [0], [coarse_label_bytes]), tf.int32)
    fine_label = tf.cast(tf.strided_slice(record_bytes, [coarse_label_bytes], [coarse_label_bytes + fine_label_bytes]), tf.int32)

    if coarse_or_fine == 'fine':
        result.label = fine_label # 100个精细分类
    else:
        result.label = coarse_label #20个粗略分类

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [coarse_label_bytes + fine_label_bytes], [coarse_label_bytes + fine_label_bytes + image_bytes]),
                             [result.depth, result.height, result.width]) # 这种顺序，是Caffee的顺序
    # 将图像的空间位置和深度位置由[depth, height, width]转换成[height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result

# 即可以读取训练集，也可以读取测试集，但是一般用来读取测试集,也就是eval_data一般是为True的
def inputs(cifar10or20or100, eval_data, data_dir, batch_size, image_size = 32):
    if cifar10or20or100 == 10:
        read_cifar = read_cifar10
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i) for i in xrange(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    elif cifar10or20or100 == 20 or cifar10or20or100 == 100:
        read_cifar = read_cifar100
        if not eval_data:
            filenames = [os.path.join(data_dir, 'train.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    if cifar10or20or100 ==  10:
        coarse_or_fine = None
    elif cifar10or20or100 == 20:
        coarse_or_fine = 'fine'
    else:
        coarse_or_fine = 'coarse'

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar(filename_queue, coarse_or_fine=coarse_or_fine)
    casted_image = tf.cast(read_input.uint8image, tf.float32)

    height = image_size
    width = image_size

    resized_image = tf.image.resize_image_with_crop_or_pad(casted_image, width, height)

    #数据集标准化操作：减去均值 + 方差归一化
    float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, IMAGE_DEPTH])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)

# 读取训练集，需要扩充数据，比如对比度变换，随机扭曲，噪声，
def distorted_inputs(cifar10or20or100, data_dir, batch_size, image_size = 32):
    if cifar10or20or100 == 10:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
        read_cifar = read_cifar10
        coarse_or_fine = None
    elif cifar10or20or100 == 20:
        filenames = [os.path.join(data_dir, 'train.bin')]
        read_cifar = read_cifar100
        coarse_or_fine = 'coarse'
    else:
        filenames = [os.path.join(data_dir, 'train.bin')]
        read_cifar = read_cifar100
        coarse_or_fine = 'fine'

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar(filename_queue, coarse_or_fine)

    casted_image = tf.cast(read_input.uint8image, tf.float32)

    height = image_size
    width = image_size

    padd_image = tf.image.resize_image_with_crop_or_pad(casted_image, width + 4, height + 4)

    # 在[36,36]大小的图像中随机裁剪出[height, width]即[32,32]的图像区域
    distorted_image = tf.random_crop(padd_image, [height, width, 3])

    # 调整亮度和对比度两个操作是不可以交换的，也就是说先调整亮度然后调整对比度，与先调整对比度再调整亮度，结果是不一样的
    distorted_image = tf.image.random_flip_left_right(distorted_image) # 水平翻转
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63) #调整亮度，每个像素点都加上一个数字，范围在-63~63之间
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8) # 对比度，每个像素点都乘上一个数字

    #数据集标准化：去均值 + 方差归一化
    float_image = tf.image.per_image_standardization(distorted_image)

    float_image.set_shape([height, width, IMAGE_DEPTH])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. THis will take a few minutes.' % min_queue_examples)
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

# 构造样本队列，产生一个批次的图像和标签，image只是一个图像
# 创建一个队列，随机打乱样本，然后读取batch_size这么多批次的图片和标签
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads=num_preprocess_threads,
                                                     capacity=min_queue_examples + 3 * batch_size,
                                                     min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads,
                                             capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images, max_outputs=9)

    return images, tf.reshape(label_batch, [batch_size])