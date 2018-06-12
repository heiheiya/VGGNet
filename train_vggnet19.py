import tensorflow as tf
from vggnet19 import VGGNet19
import math
import numpy as np
from datetime import datetime
import util

image_size = 224
num_channels = 3
num_classes = 1000

drop_rate = 0.5
batch_size = 32
num_batches = 100

def run_benchmark():
    with tf.Graph().as_default():
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, num_channels], dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        model = VGGNet19(images, keep_prob, num_classes, batch_size, image_size, num_channels)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            util.time_tensorflow_run(sess, num_batches, model.predictions, {keep_prob: 1.0}, "Forward")
            objective = tf.nn.l2_loss(model.fc8)
            grad = tf.gradients(objective, model.p)
            util.time_tensorflow_run(sess, num_batches, grad, {keep_prob:drop_rate}, "Forward-backward")

run_benchmark()