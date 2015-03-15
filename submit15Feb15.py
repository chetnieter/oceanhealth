#
# Script to generate submission for OceanHealth Kaggle contest
#

import ImageImport
import FeatureExtraction
import Utils

from sklearn import svm

print "Importing training data"
classNames, y, X = ImageImport.LoadTrainingData()

print "Importing test data"
imageNames, X_test = ImageImport.LoadTestData()

print "Training SVM at C = 0.25"
mySVM = svm.SVC(probability=True, C=0.25)
mySVM.fit(X,y)

print "Making predictions"
y_pred = mySVM.predict_proba(X_test)

print "Generating submission file"
Utils.WriteSubmission('OceanHealth14Feb15c0.25.csv', imageNames, classNames, y_pred)

print "Training SVM at C = 0.5"
mySVM = svm.SVC(probability=True, C=0.5)
mySVM.fit(X,y)

print "Making predictions"
y_pred = mySVM.predict_proba(X_test)

print "Generating submission file"
Utils.WriteSubmission('OceanHealth14Feb15c0.50.csv', imageNames, classNames, y_pred)

print "Training SVM at C = 0.75"
mySVM = svm.SVC(probability=True, C=0.75)
mySVM.fit(X,y)

print "Making predictions"
y_pred = mySVM.predict_proba(X_test)

print "Generating submission file"
Utils.WriteSubmission('OceanHealth14Feb15c0.75.csv', imageNames, classNames, y_pred)