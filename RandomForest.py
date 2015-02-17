#
# Train Random Forest
#

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

def trainRandomForest(X,y):
    print "Training"
    # n_estimators is the number of decision trees
    # max_features also known as m_try is set to the default value of the square root of the number of features
    clf = RF(n_estimators=100, n_jobs=3);
    print "Random trained"
    return clf