**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image01]: ./output_images/0.cars.visualize.jpg
[image02]: ./output_images/0.notcars.visualize.jpg
[image1]: ./output_images/1.hog.test.jpg
[image20]: ./output_images/2.classifier.training.color.jpg
[image21]: ./output_images/2.classifier.training.hog.jpg
[image3]: ./output_images/3.slidewindow.test.jpg
[image4]: ./output_images/4.multiscalewindow.test.jpg
[image5]: ./output_images/5.search.test.jpg
[image6]: ./output_images/6.hogsubsampling.test.jpg
[image7]: ./output_images/7.heatmap.jpg
[image8]: ./output_images/8.test.all.jpg
[video1]: ./result.project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

The code is based on the lesson´s code and quizes.
You can find the python files: `train_classifier.py`, `pipeline.py` and `utils.py`.
Also a txt file where I wrote down some parameters and classifier scores: `classifier.parameters.txt`
For visual reference and more tests, I used the notebook: `Project_notebook.ipynb`

### Dataset exploration

Before we get on to extracting HOG features and training a classifier, let's explore the dataset a bit before deciding what features to use, for example a combination of color and gradient features.

I downloaded first the two datasets provided at class: cars and notcars images, then I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image01]
![alt text][image02]

After checking the images from the datasets, I will continue doing some features extraction tests.

I will begin with displaying the HOG features of one single image, to see how it looks like (check `test_get_hog_features()` in the Notebook)

![alt text][image1]

Now, let´s start extracting features. First, Color features. I define two methods `bin_spatial()` and `color_hist()` based on the lessons that will create a feature vector based on color and histogram of the image. I chose the 'HLS' colorspace as I detected better results in our predictions.
Please check `test_extract_color_features()` in the Notebook.

![alt text][image20]

Then, I will check how the classifier works with HOG features only. I define the method `get_hog_features()`, and tested the predictions at `test_extract_hog_features()`. Predictions are up to 96%-98%. Not bad.

![alt text][image21]

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

After some initial exploration, I can begin with the full feature extraction. I decided to use both color and HOG features. The code is located at 'Color & HOG features combined' in the Notebook. I defined two important method `extract_features_img()` for extracting features of a single image, and `extract_features_list()` to perform the operation for a list of images.

I decided to extract HOG features from the channel Y of the colorspace YCrCb due to its emphasis on luminescence.

So I performed these steps:
* Extract features for cars images.
* Extract features for notcars images.
* Join both cars and notcars vectors, normalize and scale the total, so we work in the same data range.
* Split the data in training and test sets.
* Create the classifier (linear SVM) and train it.
* Test the classifier with the test set.
* Save the classifier to file for later uses.

#### 2. Explain how you settled on your final choice of HOG parameters.

Most of my code is based on the code provided at class, that works almost perfect out of the box. I switch to 'HLS' colorspace which I saw it provided better results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

According to the lessons learned in class, linear SVM works better with HOG features, so I tried that approach, using the color and HOG features extracted.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started using the code learned from class to perform a sliding window search. My first approach is to find around the middle part of the image, where I think most of the cars will be. This is an example.

![alt text][image3]

Please check the code 'Sliding window technique' in the Notebook.

After this, I went to try the Multi scale window search, which it will help to determine cars with different sizes depending on their size. This is an example:

![alt text][image4]

Finally, I defined the method `search_windows()` to test all this with a real example, with the help of our previous classifier. The result with a test example looks like this:

![alt text][image5]

Now, time to do some improvements, and try the `find_cars()` method proposed in class, which will help to improve speed using what is called as "Hog Sub-sampling Window Search". 

In this example, you can see the search of cars in the image with different windows sizes, and the result in the last picture.

![alt text][image6]

The last step is creating a heatmap with the boxes detected in the previous step, to compensate the overlapping of boxes detecting the same car. 

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.

![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Problems:
	* While most of the code was based on the lessons code, I found really difficult to have a decent result on the video. I achieved scores of more than 99% in my LinearSVC classifier but when applied to the video the resuls were so poor. It was more a work of trial and error than really understanding what problem was causing all this.
	* I want to add that the whole pipeline is very slow. I wondered how real cars work in real time.
	* Also the pipeline is not ready to detect cars overlapping nor occlusions. It will need a more finer approach at the CarDetection class.

