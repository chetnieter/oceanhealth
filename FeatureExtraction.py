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

def getMaxRegion(image):
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

    regionmaxprop = None
    for regionprop in region_list:
        # check to see if the region is at least 50% nonzero
        if sum(imagethr[label_list == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getAxisRatioFromRegion(region):
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not region is None) and  (region.major_axis_length != 0.0)):
        ratio = 0.0 if region is None else  region.minor_axis_length*1.0 / region.major_axis_length
    return ratio

def getAxisRatioFromImage(image):
    maxregion = getMaxRegion(image)

    return getAxisRatioFromRegion(maxregion)

def getAreaFromRegion(region, image):
    area = 0.0 if region is None else  region.area/image.size
    return area

def getAreaFromImage(image):
    maxregion = getMaxRegion(image)

    return getAreaFromRegion(maxregion, image)

def getPerimeterFromRegion(region, image):
    fullPerm = 2.*(image.shape[0] + image.shape[1])
    perm = 0.0 if region is None else  region.perimeter/fullPerm
    return perm

def getPerimeterFromImage(image):
    maxregion = getMaxRegion(image)

    return getPerimeterFromRegion(maxregion, image)

def getEulerNumFromRegion(region):
    enum = 0 if region is None else region.euler_number
    return enum

def getEulerNumFromImage(image):
    maxregion = getMaxRegion(image)

    return getEulerNumFromRegion(region)

def getFilledAreaFromRegion(region, image):
    area = 0.0 if region is None else region.filled_area/region.area
    return area

def getFilledAreaFromImage(image):
    maxregion = getMaxRegion(image)

    return getFilledAreaFromRegion(region, image)

def getConvexHullAreaFromRegion(region, image):
    area = 0.0 if region is None else region.convex_area/region.area
    return area

def getHuMomentFromRegion(region):
    huVec = (0.,0.,0.,0.,0.,0.,0.) if region is None else region.moments_hu 
    return huVec

def  getSolidityFromRegion(region):
    solidity = 0. if region is None else region.solidity
    return solidity

def getEccentricityFromRegion(region):
    eccentricity = 0. if region is None else region.eccentricity
    return eccentricity

def getInertiaEigenFromRegion(region):
    eigen = (0.,0.) if region is None else region.inertia_tensor_eigvals
    return eigen