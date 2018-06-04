import cv2
import numpy as np
from utils import get_feature_image, bin_spatial, color_hist, get_hog_features

from skimage.feature import hog

import time
from math import sqrt

from moviepy.editor import VideoFileClip
import pickle
from scipy.ndimage.measurements import label

from os import listdir
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, 
                colorspaces, spatial_size, bin_channel,
                hist_bins, hist_range, hist_channel,
                orient, pix_per_cell, cell_per_block, hog_channel,
                spatial_feat, hist_feat, hog_feat):

    window_list = []
    img = img.astype(np.float32)/255 # It´s a JPG

    img_tosearch = img[ystart:ystop,:,:]
    color_feature_img = get_feature_image(img_tosearch, colorspaces[0])
    hog_feature_img = get_feature_image(img_tosearch, colorspaces[1])
    if scale != 1:
        imshape = color_feature_img.shape
        color_feature_img = cv2.resize(color_feature_img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        hog_feature_img = cv2.resize(hog_feature_img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (color_feature_img.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (color_feature_img.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step =  2 # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
	# Compute individual channel HOG features for the entire image
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog1 = get_hog_features(hog_feature_img[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(hog_feature_img[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(hog_feature_img[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
        else:
            hog1 = get_hog_features(hog_feature_img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            features = []
            # Extract HOG for this patch
            if hog_feat == True:
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    features.append(np.hstack((hog_feat1, hog_feat2, hog_feat3)))
                else:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    features.append(np.hstack((hog_feat1)))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(color_feature_img[ytop:ytop+window, xleft:xleft+window], (window,window))
          
            # Get color features
            if spatial_feat == True:
            	features.append(bin_spatial(subimg, size=spatial_size, bin_channel=bin_channel))
            if hist_feat == True:
            	features.append(color_hist(subimg, nbins=hist_bins, bins_range=hist_range, hist_channel=hist_channel))

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:                
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # Append window position to list
                window_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

    return window_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_labeled_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    # Return the image
    return bboxes

def draw_labeled_bboxes(img, bboxes):    
    # Iterate through all detected cars
    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class CarDetection():
    def __init__(self, votes_threshold=5):
        # These list MUST have same length
        self.bboxes = []
        self.centroids = []
        self.votes = []
        self.votes_threshold = votes_threshold

    def add_bboxes(self, bboxes, threshold=10):
        prev_bboxes = self.bboxes
        prev_centroids = self.centroids
        prev_votes = self.votes
        self.bboxes = []
        self.centroids = []
        self.votes = []
        prev_found = []
    	# Add new bboxes with vote 1
        for bbox in bboxes:
            centroid = (bbox[0][0]+(bbox[1][0]/2), bbox[0][1]+(bbox[1][1]/2))
            self.centroids.append(centroid)
            self.bboxes.append(bbox)
            self.votes.append(1)

        # Check matches with  the old ones
        for ip, prev_centroid in enumerate(prev_centroids):
            match = False
            for ic, centroid in enumerate(self.centroids):
                if sqrt( (centroid[0]-prev_centroid[0])**2 + (centroid[1]-prev_centroid[1])**2 ) <= threshold:
                    # We have a match
                    match = True
                    # Promote
                    self.votes[ic] = prev_votes[ip] + 1
                    # Interpolate
                    """
                    self.bboxes[ic] = \
                        ( ( (self.bboxes[ic][0][0]+prev_bboxes[ip][0][0])//2, (self.bboxes[ic][0][1]+prev_bboxes[ip][0][1])//2),
                        ( (self.bboxes[ic][1][0]+prev_bboxes[ip][1][0])//2, (self.bboxes[ic][1][1]+prev_bboxes[ip][1][1])/2) )
                    """
                    break
            if match == False:
                # No match. Add but downvote
                vote = prev_votes[ip]-1
                if vote > 0:
                    self.centroids.append(prev_centroid)
                    self.bboxes.append(prev_bboxes[ip])
                    self.votes.append(prev_votes[ip]-1)

    def get_detections(self):
    	detections = []
    	for i, vote in enumerate(self.votes):
    		if vote > self.votes_threshold:
    			detections.append(self.bboxes[i])
    	return detections

def process_image(image, debug=False):
    windows = []
    for l in find_cars_limits:
    	ystart = l[0]
    	ystop = l[1]
    	scale = l[2]
    	windows.append(find_cars(image, ystart, ystop, scale, svc, X_scaler,
                                colorspaces, spatial_size, bin_channel,
                                hist_bins, hist_range, hist_channel,
                                orient, pix_per_cell, cell_per_block, hog_channel,
                                spatial_feat, hist_feat, hog_feat))

    hot_windows = sum(windows, [])

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    #heat = add_heat(heat, car_detection.get_hot_windows())
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    bboxes = get_labeled_bboxes(labels)

    # Update detections
    car_detection.add_bboxes(bboxes, threshold=30)
    detections = car_detection.get_detections()
    print(detections)
    draw_img = draw_labeled_bboxes(np.copy(image), detections)

    if debug == True:
        return draw_img, heatmap
    return draw_img


#### MAIN

# load a pre-trained svc model from a serialized (pickle) file
print('Loading classifier')
with open("mylinearsvc.v02.p", "rb") as pfile:
    dist_pickle = pickle.load(pfile)

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]

    spatial_size = dist_pickle['spatial_size']
    bin_channel = dist_pickle['bin_channel']
    hist_bins = dist_pickle['hist_bins']
    hist_channel = dist_pickle['hist_channel']
    colorspaces = dist_pickle['colorspaces']
    orient = dist_pickle['orient']
    pix_per_cell = dist_pickle['pix_per_cell']
    cell_per_block = dist_pickle['cell_per_block']
    hog_channel = dist_pickle['hog_channel']
    hist_range = dist_pickle['hist_range']
    spatial_feat = dist_pickle['spatial_feat']
    hist_feat = dist_pickle['hist_feat']
    hog_feat = dist_pickle['hog_feat']

find_cars_limits = []

"""
find_cars_limits.append([400, 510, 1.0])
find_cars_limits.append([380, 480, 1.2])
find_cars_limits.append([420, 550, 1.5])
find_cars_limits.append([400, 600, 2.0])
"""
"""
find_cars_limits.append([380, 444, 1.0])
find_cars_limits.append([410, 474, 1.0])
find_cars_limits.append([380, 476, 1.5])
find_cars_limits.append([420, 516, 1.5])
find_cars_limits.append([380, 508, 2.0])
find_cars_limits.append([420, 548, 2.0])
find_cars_limits.append([400, 596, 3.5])
find_cars_limits.append([464, 600, 3.5])
"""
### Windows of 64 in scale 1.0
find_cars_limits.append([400, 464, 1.0])
find_cars_limits.append([416, 480, 1.0])
### Windows of 96 in scale 1.5
find_cars_limits.append([400, 496, 1.5])
find_cars_limits.append([448, 544, 1.5])
### Windows of 128 in scale 2.0
find_cars_limits.append([400, 528, 2.0])
find_cars_limits.append([462, 590, 2.0])
### Windows of 64 in scale 3.0
find_cars_limits.append([400, 592, 3.0])
find_cars_limits.append([496, 692, 3.0])

VIDEO = False
if VIDEO == True:
    car_detection = CarDetection(5)
    videoname = 'project_video.mp4'
    video_output = 'result.v02.'+ videoname
    clip1 = VideoFileClip(videoname)
    process_clip = clip1.fl_image(process_image)
    process_clip.write_videofile(video_output, audio=False)
else:
    path = 'frames_project_video/'
    images = listdir(path)[-50:]
    for i in range(5):
        car_detection = CarDetection(0)
        ind = np.random.randint(0, len(images))
        image = mpimg.imread(path + images[ind])
        draw_img, heatmap = process_image(image, True)

        fig = plt.figure(figsize=(10, 8))
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Detections')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

