# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/exploration_01.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./writeup_images/example01.png "Traffic Sign 1"
[image5]: ./writeup_images/example02.png "Traffic Sign 2"
[image6]: ./writeup_images/example03.png "Traffic Sign 3"
[image7]: ./writeup_images/example04.png "Traffic Sign 4"
[image8]: ./writeup_images/example05.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yaboo-oyabu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. These bar charts are showing how the number of examples (y-axis) is distributed across classes (x-axis) in Train, Valid and Test datasets (graph). By seeing these charts, you can find each dataset have a similar distribution, which means is less data drift in terms of the number of examples across classes. However, it's obvious that datasets are unbalanced in terms of the number of examples between different classes, which may increase bias of a machine learning model. If a machine learning model doesn't perform well on these datasets, we should try to apply data augmentation technique or sampling method to generate balanced datasets. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized each image by very simple normalization approach, which just calculate (pixel - 128) / 128 so that the data has mean zero and equal variance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x64  			    |
| Dropout               | keep_prob = 0.8 for training, 1.0 for eval    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x128	|
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x128                   |
| Dropout               | keep_prob = 0.8 for training, 1.0 for eval    |
| Fully connected		| outputs 480x1 								|
| RELU   				|           									|
| Dropout               | keep_prob = 0.8 for training, 1.0 for eval    |
| Fully connected 		| outputs 120x1   								|
| Softmax   			|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model with AdamOptimizer and set hyperparameters as follows:

| Hyperparameter name	|     Value     	        					| 
|:---------------------:|:---------------------------------------------:| 
| BATCH SIZE      		| 128   							            | 
| EPOCHS            	| 20                                        	|
| LEARNING RATE  		| 0.001											|
| keep_prob (Dropout)   | 0.8                                           |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.949
* test set accuracy of 0.953

I chose an iterative approach:
* I started with simple LeNet architecture that is used in the lectures of self-driving car engineer nanodegree course. Validation set accuracy was less than 0.93 with this configuration.
* Increased epochs from 10 to 20. Validation set accuracy is increased a bit, but less than 0.93 with this setting.
* Added dropout layer to each convolutional layer and the first full connected layer, and set keep_prob to 0.8. The main purpose of this layer is to regularize the machine learning model.
* In addition to dropout, I increased a model complexity by increasing the number of output channels for each convolutional layers from [6, 10] to [64, 128]. This is done because I assumed the model complexity is not sufficient to detect traffic sign with higher accuracy.
* Finally, with above iterative tuning, I was able to get validation set accuracy of 0.949.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because there are many steakers on the traffic sign and ML model may be confused by them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| General caution		| General caution								|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Stop	      		    | Speed limit (60km/h)  		        		|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.953.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is defined in `predict` method. located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a Road work sign (probability of 1.00), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Road work   									| 
| 5.2703707e-17			| Pedestrians						            |
| 1.8756635e-17			| Speed limit (80km/h)				            |
| 5.5470525e-18			| Dangerous curve to the right          		|
| 8.1420654e-19		    | Right-of-way at the next intersection         |

For the second image, the model is sure that this is a General caution sign(probability of 1.00), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| General caution   							| 
| 4.2048968e-28			| Traffic signals          		    			|
| 0.0000000e+00			| Speed limit (20km/h)              			|
| 0.0000000e+00         | Speed limit (30km/h)  		 				|
| 0.0000000e+00		    | Speed limit (50km/h)                   		|

For the third image, the model is sure that this is a Speed limit (30km/h) sign(probability of 1.00), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00			| Speed limit (30km/h)							| 
| 2.6337520e-17			| Speed limit (70km/h)							|
| 3.6019260e-20			| Speed limit (50km/h)            				|
| 3.5539629e-21         | Speed limit (20km/h)    		 				|
| 8.9600170e-22		    | Speed limit (80km/h)                    		|

For the forth image, the model is sure that this is a Speed limit (30km/h) sign(probability of 1.00), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9780089e-01			| Speed limit (30km/h)							| 
| 2.1989816e-03			| Speed limit (50km/h)							|
| 9.0197311e-08			| Speed limit (70km/h)            				|
| 9.3230128e-12         | Speed limit (70km/h)    		 				|
| 9.0824310e-12		    | Roundabout mandatory                   		|

For thr fifth image, the model is relatively sure that this is a Speed limit (60km/h) sign(probability of 0.78). However, the image doesn't contain Speed limit sign, and it actually is Stop sign. I guess this is caused by many steakers put on the Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.8317517e-01			| Speed limit (60km/h) 							| 
| 1.5682380e-01			| Speed limit (80km/h)							|
| 4.7169115e-02			| Speed limit (50km/h)             				|
| 7.3429467e-03         | Stop                  		 				|
| 2.3459226e-03         | Speed limit (120km/h)    		 				|