#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature

choice = raw_input('If you want a feature list of bonus and salary press 1 else if u want a feature list of bonus and long_term_incentive press 2 =')
features_list=[]
fit_test=False
if choice=='1':
    features_list= ["bonus", "salary"]
    fit_test_for_outlier=raw_input('do you want to fit the line for the outliers say "yes" =')
    if fit_test_for_outlier=='yes':
        fit_test=True
else:
    features_list=["bonus" ,"long_term_incentive"]

#split the numpy array in line 40
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
print data.shape
print data
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.


sys.path.append("C:/Users/Usman_1/ud120-projects-master/ud120-projects-master/tools/")
sys.path.append('C:/Users/Usman_1/ud120-projects-master/ud120-projects-master/choose_your_own')
sys.path.append('C:/Users/Usman_1/ud120-projects-master/ud120-projects-master/datasets_questions')

import os
os.chdir('C:/Users/Usman_1/ud120-projects-master/ud120-projects-master/regression')
#
# import numpy
# import matplotlib
# matplotlib.use('agg')
# from sklearn import linear_model
#
# def studentReg(feature_train, target_train):
#     reg = linear_model.LinearRegression()
#     reg.fit(feature_train, target_train)
#     return reg
#
#
# reg = studentReg(feature_train, target_train)

# plt.clf()
# plt.scatter(feature_train, target_train, color="b", label="train data")
# plt.scatter(feature_test, target_test, color="r", label="test data")
# plt.plot(feature_test, reg.predict(feature_test), color="black")
# plt.legend(loc=2)
# plt.xlabel("ages")
# plt.ylabel("net worths")
#





from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)



### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color )
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color )

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()

if fit_test:
        reg.fit(feature_test, target_test)
        plt.plot(feature_train, reg.predict(feature_train), color="r")
        plt.show()
else:
    plt.show()

#Extracting Slope and Intercept
print 'slope = {0}'.format(reg.coef_[0])
print 'intercept = {0}'.format(reg.intercept_)


#Regression Score: Training Data
print 'score on training set = {0}'.format(reg.score(feature_train, target_train))

#Regression Score: Test Data
print 'score on test set = {0}'.format(reg.score(feature_test, target_test))
#slope
print 'slope = {0}'.format(reg.coef_[0])