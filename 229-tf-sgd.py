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
x = tf.placeholder(tf.float32, [None, 50])
W = tf.Variable(tf.zeros([50, 8]))
b = tf.Variable(tf.zeros([8]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32,[None, 8])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(init)

for i in range(15000):
  # batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x:X_train, y_:y_train})

print(sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))

