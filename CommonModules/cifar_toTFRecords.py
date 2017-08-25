import pickle
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 #训练集
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 #测试集，评估集
IMAGE_SIZE = 32
IMAGE_DEPTH = 3

def _read_data_batches_py(data_dir, train=True, cifar10or20or100=10):
    if cifar10or20or100 == 10:
        if train:
            batches = [pickle.load(open(os.path.join(data_dir, 'data_batch_%d' % i), 'rb'), encoding='bytes') for i in range(1, 6)]
        else:
            batches = [pickle.load(open(os.path.join(data_dir, 'test_batch'), 'rb'), encoding='bytes')]
    else:
        if train:
            batches = [pickle.load(open(os.path.join(data_dir, 'train'), 'rb'), encoding='bytes')]
        else:
            batches = [pickle.load(open(os.path.join(data_dir, 'test'), 'rb'), encoding='bytes')]

    images = np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype=np.uint8) if train else np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype=np.uint8)
    labels = np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN), dtype=np.int32) if train else np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_EVAL), dtype=np.int32)
    for i, b in enumerate(batches):
        if cifar10or20or100 == 10:
            for j, l in enumerate(b[b'labels']):
                images[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = b[b'data'][j].reshape([IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = l
        elif cifar10or20or100 == 20:
            for j, l in enumerate(b[b'coarse_labels']):
                images[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = b[b'data'][j].reshape([IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = l
        else:
            for j, l in enumerate(b[b'fine_labels']):
                images[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = b[b'data'][j].reshape([IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = l
    return images, labels

def _convertToTFRecords(images, labels, num_examples, data_dir, filename):
    """
    Args:
        images: (num_examples, height, width, channels) np.int64 nparray (0~255)
        labels: (num_examples) np.int64 nparray
        num_examples: number of examples
        filename: the tfrecords' name to be saved
    Return: None, but store a .tfrecords file to data_log/
    """
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, filename))
    for index in xrange(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def readFromTFRecords(filename, batch_size, img_shape, num_threads=2, min_after_dequeue=1000):
    def read_and_decode(filename_queue, img_shape):
        """Return a single example for queue"""
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        # some essential steps
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, img_shape)    # THIS IS IMPORTANT
        image.set_shape(img_shape)
        image = tf.cast(image, tf.float32) * (1 / 255.0)  # set to [0, 1]

        sparse_label = tf.cast(features['label'], tf.int32)

        return image, sparse_label

    filename_queue = tf.train.string_input_producer([filename])

    image, sparse_label = read_and_decode(filename_queue, img_shape) # share filename_queue with multiple threads

    # tf.train.shuffle_batch internally uses a RandomShuffleQueue
    images, sparse_labels = tf.train.shuffle_batch(
        [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
        min_after_dequeue=min_after_dequeue,
        capacity=min_after_dequeue + (num_threads + 1) * batch_size
    )

    return images, sparse_labels

if __name__ == '__main__':
    dataset_dir = '../Total_Data/TempData/cifar_tfrecords'
    cifar10or20or100 = [[10,'../Total_Data/TempData/cifar-10-batches-py'],
                        [20,'../Total_Data/TempData/cifar-100-python'],
                        [100,'../Total_Data/TempData/cifar-100-python']]
    train = [True, False]

    for i, cifar in enumerate(cifar10or20or100):
        for j, t in enumerate(train):
            if t:
                filename = 'train_%i_package.tfrecords' % cifar[0]
            else:
                filename = 'test_%i_package.tfrecords' % cifar[0]

            if not os.path.exists(os.path.join(cifar[1], filename)):
                images, labels = _read_data_batches_py(cifar[1], t, cifar[0])
            _convertToTFRecords(images, labels, len(images), dataset_dir, filename)
            print(filename + ' done')