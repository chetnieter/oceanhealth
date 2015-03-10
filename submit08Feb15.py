#
# Script to generate submission for OceanHealth Kaggle contest
#

import ImageImport
import FeatureExtraction
import Utils

from sklearn.ensemble import RandomForestClassifier as RF

print "Importing training data"
classNames, y, X = ImageImport.LoadTrainingData()

print "Training random forest"
myRF = RF(n_estimators=200)
myRF.fit(X,y)

print "Importing test data"
imageNames, X_test = ImageImport.LoadTestData()

print "Making predictions"
y_pred = myRF.predict_proba(X_test)

print "Generating submission file"
Utils.WriteSubmission('OceanHealth08Feb15.csv', imageNames, classNames, y_pred)