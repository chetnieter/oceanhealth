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

# make graphics inline
# %matplotlib inline

pathToData = "..\data"

# get the classnames from the directory structure
# directory_names = list(set(glob.glob(os.path.join(pathToData,"train", "*")) \
#  ).difference(set(glob.glob(os.path.join(pathToData,"train","*.*")))))
directory_names = glob.glob(os.path.join(pathToData,"train", "*"))

# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
example_file = glob.glob(os.path.join(directory_names[0],"*.jpg"))[9]
print example_file
im = imread(example_file, as_grey=True)
# plt.imshow(im, cmap=cm.gray)
# plt.show()

# First we threshold the image by only taking values greater than the mean to reduce noise in the image
# to use later as a mask
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
imdilated = morphology.dilation(imthr, np.ones((4,4)))

labels = measure.label(imdilated)
labels = imthr*labels
labels = labels.astype(int)

regions = measure.regionprops(labels)