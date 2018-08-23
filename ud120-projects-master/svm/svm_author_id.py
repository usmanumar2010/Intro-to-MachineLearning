#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from nltk.chunk.util import accuracy
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine
# https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from time import time
from email_preprocess import preprocess
import numpy as np

#https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
#https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


def my_svm(features_train, features_test, labels_train, labels_test, kernel='linear', C=1.0):
    # the classifier
    clf = SVC(kernel=kernel, C=C)

    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time() - t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time() - t0, 3), "s"

    accuracy = accuracy_score(pred, labels_test)

    print '\naccuracy = {0}'.format(accuracy)
    return pred


# pred = my_svm(features_train, features_test, labels_train, labels_test)


# print len(features_train)  #15820
# print features_train
# print  len(features_train)/100   #15820/100=158.2=158


# A Smaller Training Set
features_train_2 = features_train[:len(features_train) / 100]
labels_train_2 = labels_train[:len(labels_train) / 100]
# pred=my_svm(features_train_2,features_test,labels_train_2,labels_test)

# Deploy an RBF Kernel

# pred = my_svm(features_train_2, features_test, labels_train_2, labels_test, 'rbf')



# Optimize C Parameter
print 'Optimize C Parameter'

# for C in [10,100,1000,10000]:
#     print 'C =',C,
#     pred=my_svm(features_train_2, features_test, labels_train_2, labels_test, 'rbf',C=C)
#     print '\n\n'
#

# Optimized RBF vs. Linear SVM: Accuracy
print 'RBF vs. Linear SVM: Accuracy'

pred = my_svm(features_train, features_test, labels_train, labels_test, kernel='rbf', C=10000)

#Extracting Predictions from an SVM
print 'Extracting Predictions from an SVM'
print pred[10]
print pred[26]
print pred[50]

#How many Chris emails predicted?
print 'How many Chris emails predicted?'
print sum(pred)

#########################################################
