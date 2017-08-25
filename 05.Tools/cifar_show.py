import numpy as np
import matplotlib.pyplot as plt
import cifar_input
import tensorflow as tf
from PIL import Image
cifar10or20or100 = 10
dataset_dir = '../Total_Data/Cifar10_data/cifar-10-batches-bin'
image, label = cifar_input.images(cifar10or20or100, eval_data=True, batch_size=1, data_dir=dataset_dir)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners()
    # for idx in range(10):
    image_batch, label_batch = sess.run([image, label])

    plt.imshow(image_batch)
    plt.show()
