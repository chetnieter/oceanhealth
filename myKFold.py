#
# Run K-Fold validation and training to get predictions
#

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

#
# Run KFold to produce absolute predictions
#
# X features vector
# y vector with class id's
# cls Classifier from scikit learn 
# 
def runKFoldAbs(X, y, cls, n_folds=5):
    kf = KFold(y, n_folds)
    y_pred = y * 0

    for train, test in kf:
        print "Generating test and train data for fold"
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
        print "Fitting data for fold"
        cls.fit(X_train, y_train)
        print "Running prediction for fold"
        y_pred[test] = cls.predict(X_test)

    return y_pred

#
# Run KFold to produce probablistic predictions
#
# X features vector
# y vector with class id's
# cls Classifier from scikit learn 
# 
def runKFoldProb(X, y, cls, n_folds=5):
    kf = KFold(y, n_folds)
    y_pred = np.zeros((len(y),len(set(y))))

    for train, test in kf:
        print "Generating test and train data for fold"
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
        print "Fitting data for fold"
        cls.fit(X_train, y_train)
        print "Running prediction for fold"
        y_pred[test] = cls.predict_proba(X_test)

    return y_pred