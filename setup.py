#
# Imports libraries needed and set up data structures used for training and
# analysis
#

#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
from Utils import *
import FeatureExtraction as FE

pathToData = "..\data"

# get the classnames from the directory structure
# directory_names = list(set(glob.glob(os.path.join(pathToData,"train", "*")) \
#  ).difference(set(glob.glob(os.path.join(pathToData,"train","*.*")))))
directory_names = glob.glob(os.path.join(pathToData,"train", "*"))

# Rescale the images and create the combined metrics and training labels

#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 6 # for our ratio

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            maxRegion = FE.getMaxRegion(image)
            axisratio = FE.getAxisRatioFromRegion(maxRegion)
            arearatio = FE.getAreaFromRegion(maxRegion, image)
            perimeter = FE.getPerimeterFromRegion(maxRegion, image)
            filledarea = FE.getFilledAreaFromRegion(maxRegion, image)
            convexhull = FE.getConvexHullAreaFromRegion(maxRegion, image)
            eulernum = FE.getEulerNumFromRegion(maxRegion)
            image = resize(image, (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the axis ratio
            X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
            X[i, imageSize] = axisratio
            X[i, imageSize+1] = arearatio
            X[i, imageSize+2] = perimeter
            X[i, imageSize+3] = filledarea
            X[i, imageSize+4] = convexhull
            X[i, imageSize+5] = eulernum
            
            # Store the classlabel
            y[i] = label
            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1