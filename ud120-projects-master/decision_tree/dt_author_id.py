#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import tree
import numpy as np
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#Excellent!  A split on bumpiness yields two children (bumpy and smooth) with an equal split of slow and fast classes.  There is zero information gain.
#########################################################
### your code goes here ###

#Your First Email DT: Accuracy
# clf = tree.DecisionTreeClassifier(min_samples_split=40)
# clf = clf.fit(features_train, labels_train)
#
# print clf.score(features_test, labels_test)



#Speeding Up Via Feature Selection 1
print features_train.shape


#Changing the Number of Features

# I made "percentile" an input argument for preprocess with default value 10# I mad
# features_train, features_test, labels_train, labels_test = preprocess(percentile=1)
# print features_train.shape


#Accuracy Using 1% of Features
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)
#########################################################


