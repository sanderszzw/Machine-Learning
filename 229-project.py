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
labels_vec = [list(i) for i in zip(*original_label)] # m = 1244,  n = 50

m_samples = len(data)
n_features = len(data[0])


print 'm_samples: ', m_samples, '  n_features: ', n_features

labels = [-1]*m_samples 
for i in range(m_samples):
    for j in range(len(labels_vec[0])):
        if labels_vec[i][j] == 1:
            labels[i] = j

X = data # m = 1244,  n = 50
y = labels # 1244-length vector with label from 0-7

###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################
###############################################################################

# n_components = 30

# # print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
# t0 = time()
# pca = RandomizedPCA(n_components=n_components).fit(X_train)
# print "done in %0.3fs" % (time() - t0)

# # eigenfaces = pca.components_.reshape((n_components, h, w))

# print "Projecting the input data on the eigenfaces orthonormal basis"
# t0 = time()
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)
# print "done in %0.3fs" % (time() - t0)


###############################################################################
###############################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
param_grid = {
         'C': [5e2, 1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train, y_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_


###############################################################################

###############################################################################
# Quantitative evaluation of the model quality on the test set

print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(X_test)
print "done in %0.3fs" % (time() - t0)
target_names = ['Banjo','Cello', 'Clarinet', 'English Horn', 'Guitar','Oboe','Trumpet','Violin']
print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(m_samples))
