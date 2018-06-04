import cv2
import numpy as np
from utils import get_feature_image, bin_spatial, color_hist, get_hog_features

from glob import glob
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def extract_features_img(img, colorspaces, spatial_size, bin_channel,
                        hist_bins, hist_range, hist_channel,
                        orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat):    
    #1) Define an empty list to receive features
    img_features = []
    if spatial_feat == True or hist_feat == True:
        #2) Apply color conversion if other than 'RGB'
        feature_image = get_feature_image(img, colorspaces[0])
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size, bin_channel=bin_channel)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, hist_channel=hist_channel)
            #6) Append features to list
            img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        feature_image = get_feature_image(img, colorspaces[1])
        hog_features = []
        if hog_channel == 'ALL':
            for i in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,i], orient, pix_per_cell, cell_per_block, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    #return img_features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_list(imgs, colorspaces, spatial_size, bin_channel,
                        hist_bins, hist_range, hist_channel,
                        orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat):
    #1) Define an empty list to receive features
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        img_features = extract_features_img(image, colorspaces, spatial_size, bin_channel,
                        hist_bins, hist_range, hist_channel,
                        orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat)
        features.append(img_features)

    # Return list of feature vectors
    return features

############# MAIN

path_vehicles = 'datasets/vehicles/'
path_non_vehicles = 'datasets/non-vehicles/'

cars = glob(path_vehicles + '**/*.png')
notcars = glob(path_non_vehicles + '**/*.png')

cars_shuffled = shuffle(cars)
not_cars_shuffled = shuffle(notcars)

# Features parameters
spatial_size = (32,32)
bin_channel = 'ALL'
hist_bins = 32
hist_channel = 'ALL'
colorspaces = ('HLS', 'YCrCb')# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
hist_range=(0.0, 1.0)
spatial_feat=False
hist_feat=False
hog_feat=True

t = time.time()

print('Car data:', len(cars))
print('Not car data:', len(notcars))

print('Extracting features')

car_features = extract_features_list(cars_shuffled, colorspaces=colorspaces, spatial_size=spatial_size, bin_channel=bin_channel,
                        hist_bins=hist_bins, hist_range=hist_range, hist_channel=hist_channel,
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features_list(not_cars_shuffled, colorspaces=colorspaces, spatial_size=spatial_size, bin_channel=bin_channel,
                        hist_bins=hist_bins, hist_range=hist_range, hist_channel=hist_channel,
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract Color&HOG features...')
print('Car features:', len(car_features))
print('Not car features:', len(notcar_features))

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

print('X:', len(X), X.shape)
print('y:', len(y), y.shape)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

print('Scaling data')
t = time.time()
# Fit a per-column scaler only on the training data
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to scale data...')

print('Using spatial binning of:', spatial_size, 'and', hist_bins,'histogram bins')
print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

print('Training classifier')
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
print('done')

# Dump the trained LinearSVC with Pickle
svc_pkl_filename = 'mylinearsvc.v02.p'
# Open the file to save as pkl file
svc_pkl = open(svc_pkl_filename, 'wb')
pickle.dump(svc, svc_pkl)
# Close the pickle instances
svc_pkl.close()

try:
    with open(svc_pkl_filename, 'wb+') as pfile:
        print('Saving to pickle file', svc_pkl_filename)
        pickle.dump(
        {
            'svc': svc,
            'scaler': X_scaler,
            # Features parameters
            'spatial_size':spatial_size,
            'bin_channel':bin_channel,
            'hist_bins':hist_bins,
            'hist_channel':hist_channel,
            'colorspaces':colorspaces,
            'orient':orient,
            'pix_per_cell':pix_per_cell,
            'cell_per_block':cell_per_block,
            'hog_channel':hog_channel,
            'hist_range':hist_range,
            'spatial_feat':spatial_feat,
            'hist_feat':hist_feat,
            'hog_feat':hog_feat
        },
        pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to ', svc_pkl_filename, ':', e)

print('done')