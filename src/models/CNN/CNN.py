# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:57:40 2020

@author: Pranav Aditya
"""

import tensorflow as tf
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



filename = "cheetah.cs.fiu.edu-110108-113008.2.blkparse"
#filename = "cheetah.1000"

df = pd.read_csv(filename, sep=' ',header = None)
df.columns = ['timestamp','pid','pname','blockNo', 'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
print(df.head())
df1=df.head()

df2=df[:10000]
scaler = MinMaxScaler(feature_range=(0, 1024), copy=True)
scaler.fit(df2['blockNo'].values.reshape(-1,1))
tdf=scaler.transform(df2['blockNo'].values.reshape(-1,1))

X, Y = list(), list()
for i in range(len(tdf)):
		# find the end of this pattern
        end_ix = i + 32
		# check if we are beyond the sequence
        if end_ix > len(tdf)-1:
            break
		# gather input and output parts of the pattern
        l = list(tdf[i:end_ix])*32
        seq_x=l
        
        seq_y= tdf[end_ix]
        X.append(np.array(seq_x))
        Y.append(seq_y)
X=np.array(X)
Y=np.array(Y)


X = X.reshape((X.shape[0], X.shape[1]))
X=X.astype('float32')
# Python optimisation variables
learning_rate = 0.0001
epochs = 30
batch_size =1000


# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from 
# mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None,1024])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 32, 32, 1])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 1])


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                                   padding='SAME')
    
    return out_layer
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
print(layer1)
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
print(layer2)
flattened = tf.reshape(layer2, [-1, 8 * 8* 64])
print(flattened)

wd1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.relu(dense_layer2)

cross_entropy = tf.nn.l2_loss(dense_layer2-y)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(Y) / batch_size)
    
    for epoch in range(epochs):
        avg_cost = 0
        
        print("in epoch ",epoch)
        for i in range(total_batch):
            
            batch_x=X[0:batch_size]
            batch_y = Y[0:batch_size]
            _, c = sess.run([optimiser, cross_entropy], 
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy,feed_dict={x: X, y:Y})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: X, y: Y}))