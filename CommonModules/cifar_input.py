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
                             [result.depth, result.height, result.width])
    # 将图像的空间位置和深度位置由[depth, height, width]转换成[height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result