# Various utilties to extract features from data

import numpy as np
from skimage import morphology
from skimage import measure
import pandas as pd
from matplotlib import pyplot as plt
import os

# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio

def getAreaRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    area = 0.0 if maxregion is None else  maxregion.area/image.size
    return area

def getPerimeter(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    fullPerm = 2.*(image.shape[0] + image.shape[1])
    perm = 0.0 if maxregion is None else  maxregion.area/fullPerm
    return perm

def TestSeparation(y,X,classNames):
# Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

    #Create a DataFrame object to make subsetting the data on the class 
    num_features = X.shape[1]
    df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})

    f = plt.figure(figsize=(30, 20))
    #we suppress zeros and choose a few large classes to better highlight the distributions.
    df = df.loc[df["ratio"] > 0]
    minimumSize = 20 
    counts = df["class"].value_counts()
    largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
    # Loop through 40 of the classes 
    for j in range(0,40,2):
        subfig = plt.subplot(4, 5, j/2 +1)
        # Plot the normalized histograms for two classes
        classind1 = largeclasses[j]
        classind2 = largeclasses[j+1]
        n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
                             alpha=0.5, bins=[x*0.01 for x in range(100)], \
                             label=classNames[classind1].split(os.sep)[-1], normed=1)

        n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
                              alpha=0.5, bins=bins, label=classNames[classind2].split(os.sep)[-1],normed=1)
        subfig.set_ylim([0.,10.])
        plt.legend(loc='upper right')
        plt.xlabel("Width/Length Ratio")

    plt.show()

def multiclassLogLoss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss