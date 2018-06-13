import tensorflow as tf
import math
import numpy as np
from datetime import datetime
import time

def conv_layer(input, kh, kw, n_out, sh, sw, name, p, b_value=0.0, padding='VALID'):
    with tf.variable_scope(name) as scope:
        n_in = int(input.get_shape()[-1])
        filter = tf.get_variable(name="weights", shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable(name="biases", shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(b_value), trainable=True)

        conv = tf.nn.conv2d(input=input, filter=filter, strides=[1, sh, sw, 1], padding=padding)

        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)
        p += [filter, biases]
        print_activations(relu)
        return relu


def max_pool_layer(input, kh, kw, sh, sw, name, padding='VALID'):
    max_pool = tf.nn.max_pool(input, ksize=[1, kh, kw, 1], strides=[1, sh, sw, 1], padding=padding, name=name)
    print_activations(max_pool)
    return max_pool

def full_connected_layer(input, n_out, name, p, b_value =0.0):
    shape = input.get_shape().as_list()
    dim = 1
    for d in range(len(shape)-1):
        dim *= shape[d+1]
    x = tf.reshape(input, [-1, dim])
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name="weights", shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases", shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(b_value), trainable=True)

        fc = tf.nn.relu_layer(x, weights, biases, name=scope.name)

        p += [weights, biases]
        print_activations(fc)
        return fc

def dropout(x, keep_prob, name):
    return tf.nn.dropout(x, keep_prob, name=name)

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def time_tensorflow_run(session, num_batches, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i%10:
                print('%s: step %d, duration = %.3f'%(datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared/num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch'%(datetime.now(), info_string, num_batches, mn, sd))
