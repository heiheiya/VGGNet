import numpy as np
import tensorflow as tf
import os
from vggnet16 import VGGNet16
from imagenet_classes import class_names
from scipy.misc import imread, imresize

image_size = 224
num_channels = 3
num_classes = 1000

drop_rate = 0.5
batch_size = 32
num_batches = 100

filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoint"
checkpoint_name = os.path.join(checkpoint_path, 'model.ckpt')

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

img = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels])

model = VGGNet16(img, drop_rate, num_classes, batch_size, image_size, num_channels)
prob = model.softmax

image = imread('/home/tj/work/code/tensorflow/AlexNet_V2/data/train/00004.jpg', mode='RGB')
image = imresize(image, (image_size, image_size))

with tf.Session() as sess:
    #model.load_weights("./vgg16_weights.npz", sess)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_name)
    probs = sess.run(prob, feed_dict={img: [image]})[0]
    preds = (np.argsort(probs)[::-1])[0:5]
    for p in preds:
        print(class_names[p], probs[p])