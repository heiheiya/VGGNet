import tensorflow as tf
import numpy as np
import util

class VGGNet19(object):
    def __init__(self, x, keep_prob, num_classes, batch_size, image_size, channels):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.CHANEELS = channels

        self.create()

    def create(self):
        self.p = []
        self.X = tf.reshape(self.X, shape=[-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANEELS])
        conv1_1 = util.conv_layer(self.X, kh=3, kw=3, n_out=64, sh=1, sw=1, name='conv1_1', p=self.p, padding='SAME')
        conv1_2 = util.conv_layer(conv1_1, kh=3, kw=3, n_out=64, sh=1, sw=1, name='conv1_2', p=self.p, padding='SAME')
        pool1 = util.max_pool_layer(conv1_2, kh=2, kw=2, sh=2, sw=2, name='pool1')

        conv2_1 = util.conv_layer(pool1, kh=3, kw=3, n_out=128, sh=1, sw=1, name='conv2_1', p=self.p, padding='SAME')
        conv2_2 = util.conv_layer(conv2_1, kh=3, kw=3, n_out=128, sh=1, sw=1, name='conv2_2', p=self.p, padding='SAME')
        pool2 = util.max_pool_layer(conv2_2, kh=2, kw=2, sh=2, sw=2, name='pool2')

        conv3_1 = util.conv_layer(pool2, kh=3, kw=3, n_out=256, sh=1, sw=1, name='conv3_1', p=self.p, padding='SAME')
        conv3_2 = util.conv_layer(conv3_1, kh=3, kw=3, n_out=256, sh=1, sw=1, name='conv3_2', p=self.p, padding='SAME')
        conv3_3 = util.conv_layer(conv3_2, kh=3, kw=3, n_out=256, sh=1, sw=1, name='conv3_3', p=self.p, padding='SAME')
        conv3_4 = util.conv_layer(conv3_3, kh=3, kw=3, n_out=256, sh=1, sw=1, name='conv3_4', p=self.p, padding='SAME')
        pool3 = util.max_pool_layer(conv3_4, kh=2, kw=2, sh=2, sw=2, name='pool3')

        conv4_1 = util.conv_layer(pool3, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv4_1', p=self.p, padding='SAME')
        conv4_2 = util.conv_layer(conv4_1, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv4_2', p=self.p, padding='SAME')
        conv4_3 = util.conv_layer(conv4_2, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv4_3', p=self.p, padding='SAME')
        conv4_4 = util.conv_layer(conv4_3, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv4_4', p=self.p, padding='SAME')
        pool4 = util.max_pool_layer(conv4_4, kh=2, kw=2, sh=2, sw=2, name='pool4')

        conv5_1 = util.conv_layer(pool4, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv5_1', p=self.p, padding='SAME')
        conv5_2 = util.conv_layer(conv5_1, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv5_2', p=self.p, padding='SAME')
        conv5_3 = util.conv_layer(conv5_2, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv5_3', p=self.p, padding='SAME')
        conv5_4 = util.conv_layer(conv5_3, kh=3, kw=3, n_out=512, sh=1, sw=1, name='conv5_4', p=self.p, padding='SAME')
        pool5 = util.max_pool_layer(conv5_4, kh=2, kw=2, sh=2, sw=2, name='pool5')

        fc6 = util.full_connected_layer(pool5, n_out=4096, name='fc6', p=self.p)
        fc6_dropout = util.dropout(fc6, self.KEEP_PROB, name='fc6_drop')

        fc7 = util.full_connected_layer(fc6_dropout, n_out=4096, name='fc7', p=self.p)
        fc7_dropout = util.dropout(fc7, self.KEEP_PROB, name='fc7_drop')

        self.fc8 = util.full_connected_layer(fc7_dropout, self.NUM_CLASSES, name='fc8', p=self.p)
        self.softmax = tf.nn.softmax(self.fc8)
        self.predictions = tf.argmax(self.softmax, 1)