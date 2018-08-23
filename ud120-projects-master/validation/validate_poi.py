#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import datasets

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

# iris = datasets.load_iris()
# features = iris.data
# labels = iris.target

### set the random_state to 0 and the test_size to 0.4 so
### we can exactly check your result
# from sklearn.svm import SVC
from sklearn import cross_validation
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
#
#
# clf = SVC(kernel="linear", C=1.)
# clf.fit(features_train, labels_train)
#
# print clf.score(features_test, labels_test)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features,labels)

print clf.score(features,labels)


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)


# How many POIs are in the test set for your POI identifier?
pred = clf.predict(features_test)
sum(pred)
print len([e for e in labels_test if e == 1.0])

# How many people total are in your test set?
print len(pred)

# If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
print 1.0 - 5.0/29

from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

print precision_score(labels_test, pred)

print recall_score(labels_test, pred)

print classification_report(labels_test, pred)

#How Many True Positives?

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

cm = confusion_matrix(true_labels, predictions)



print cm, '\n'
print '{0} True positives'.format(cm[1][1])
print '{0} True negatives'.format(cm[0][0])
print '{0} False positives'.format(cm[0][1])
print '{0} False negatives'.format(cm[1][0])

#Precision

print precision_score(true_labels, predictions)

#Recall
print recall_score(true_labels, predictions)