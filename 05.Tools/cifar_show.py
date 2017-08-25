import cifar_toTFRecords
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

## cifar图片的展示有两种方式
## 从cifar-10-batches-py里面直接读取

# dataset_dir = '../Total_Data/TempData/cifar-10-batches-py'
# filenames = [os.path.join(dataset_dir, 'data_batch_%' % i ) for i in range(1,6)]
# for idx, names in enumerate(filenames):
#     print(names)
#     dict = pickle.load(open(names, 'rb'), encoding='bytes')
#     X = dict[b'data']
#     Y = dict[b'labels']
#     X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#     Y = np.array(Y)
#
#     fig, axes1 = plt.subplots(5,5,figsize=(3,3))
#     for j in range(5):
#         for k in range(5):
#             axes1[j][k].set_axis_off()
#             axes1[j][k].imshow(X[j * 5 + k])
#     plt.show()

filename = 'train_package.tfrecords'
shape = [32, 32, 3]
images, labels = cifar_toTFRecords.readFromTFRecords(os.path.join('../Total_Data/cifar-10-batches-tfrecords',
                                                                  filename), 25, shape)
with tf.Session() as sess:
    tf.train.start_queue_runners()
    image_batch, _ = sess.run([images, labels])
    print(image_batch[0])
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(image_batch[j * 5 + k])
    plt.show()

