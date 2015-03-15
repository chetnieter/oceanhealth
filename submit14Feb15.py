#
# Script to generate submission for OceanHealth Kaggle contest
#

import ImageImport
import FeatureExtraction
import Utils

from sklearn import svm

print "Importing training data"
classNames, y, X = ImageImport.LoadTrainingData()

print "Training random forest"
mySVM = svm.SVC(probability=True)
mySVM.fit(X,y)

print "Importing test data"
imageNames, X_test = ImageImport.LoadTestData()

print "Making predictions"
y_pred = mySVM.predict_proba(X_test)

print "Generating submission file"
Utils.WriteSubmission('OceanHealth14Feb15.csv', imageNames, classNames, y_pred)
