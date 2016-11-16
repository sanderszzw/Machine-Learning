from time import time
import logging
import pylab as pl
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import scipy.io as sio

#read preprocess data from matlab
mat_data = sio.loadmat('1-dataset_avgvalue_chunk50.mat')
mat_label = sio.loadmat('label-50.mat')
original_input =  mat_data["input_data"]
original_label = mat_label['mapped_label_data']


# transform the shape
data = [list(i) for i in zip(*original_input)] # m = 1244,  n = 50
labels_vec = [list(i) for i in zip(*original_label)] # m = 1244,  n = 8

m_samples = len(data)
n_features = len(data[0])


print 'm_samples: ', m_samples, '  n_features: ', n_features

labels = [-1]*m_samples 
for i in range(m_samples):
    for j in range(len(labels_vec[0])):
        if labels_vec[i][j] == 1:
            labels[i] = j

X = data # m = 1244,  n = 50
Y = labels # 1244-length vector with label from 0-7

##################################################################
X_train, X_test, y_train, y_test = train_test_split(X, labels_vec, test_size=0.25, random_state=42)

import tensorflow as tf

# number 1 to 10 data

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 50]) 
ys = tf.placeholder(tf.float32, [None, 8])

# add output layer
l1 = add_layer(xs, 50, 30, activation_function = tf.nn.softmax)
prediction = add_layer(l1, 30, 8,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_train)))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(10000):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
    if i % 100 == 0:
        print 'iteration: ', i, 'accuracy: ', (compute_accuracy(
            X_test, y_test))