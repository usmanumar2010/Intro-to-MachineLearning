#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from time import time

#What sys.path.append doing ?Actually it adding the path of the folder tools where there is a email_preprocess script ,u can see the below link too
#https://jefflirion.github.io/udacity/Intro_to_Machine_Learning/Lesson1.html 
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#classifier
clf=GaussianNB()

#train
t0= time()
clf.fit(features_train,labels_train)
print "training time" ,round(time()-t0,3), 's'

#predict
#predict function use to Perform classification on an array of test vectors X.
#Predicted target values for X
t0=time()
#target labels for features of test_set will come the variable pred which will compare later to find the accuracy
pred=clf.predict(features_test)
print "predicting time" ,round(time()-t0,3), 's'

print '*****'
print pred

accuracy= accuracy_score(pred,labels_test)

print '\naccuracy = {0}'.format(accuracy)


#########################################################


