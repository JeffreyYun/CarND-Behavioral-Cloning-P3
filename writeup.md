# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./examples/nvidia-cnn-architecture.png "Model Visualization, from Nvidia blog post"
[center]: ./examples/center.jpg "Center driving"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup.md` summarizing the results
* `run1.mp4`, a recording of my vehicle driving autonomously around the track for at least one full lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network based on the Udacity lectures, Q&A, and mentioned NVIDIA CNN (["End-to-End Deep Learning for Self-Driving Cars"](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)). It consists of five convolutional layers and three fully-connected layers.

The data is first normalized in the model using a Keras Lambda layer. Then a Keras Cropping2D layer crops the images to remove the noisy scenery (top of image) and car hood (bottom of image), only keeping the road terrain.

The model uses a RELU layer after each Keras Conv2D layer to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model originally contained several dropout layers in order to reduce overfitting, one after each convolutional layer (drop prob of 0.1-0.3) and after each fully-connected layer (drop prob of 0.5). That was the idea per https://stackoverflow.com/a/47959567/6293259 and https://stackoverflow.com/a/47959567/6293259, but the final model is simplified from this, as the model was not converging quickly enough from the training data on crucial turns.

The final model architecture submitted here uses two dropout layers after two hidden fully-connected layers.

The model was trained and validated on disjunct data sets to ensure that the model was not overfitting. These were split using `sklearn.model_selection.train_test_split`. There is no specific test set; the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Proof is provided by the video.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually, for the most part. Later training used fine-tuning to incorporate new turn data without upsetting model performance on straight sections (e.g. feed Adam with 1e-4 learning rate).

Due to GPU Out of Memory errors at points, I changed batch size to 16.

I increased my steer_correction factor for left/right images, since my model was going straight too often, so I thought a more extreme steering adjustment might help (also I originally tested at higher speeds than my training data, and the car was not steering enough at certain turns)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I started with some data of center lane driving, then recorded much data recovering from the left and right sides of the road, especially in the areas the autonomous driving model failed in.

For details about how I created the training data, see the next section.

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the suggested architectures and improve upon them.

My first step was to use a convolutional neural network model similar to the NVIDIA model mentioned previously. I thought this model might be appropriate because it was used and regarded as effective for their similar subject of self-driving car behavioral cloning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This gap between loss values implied that the model was overfitting. To combat this overfitting, I modified the model to add dropout layers after the conv and fully-connected layers. Then I trained this new model with my training data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, including the brown turn after the bridge. (Getting past this turn was by far the most time-consuming step...) To improve the driving behavior in these cases, I created more training data at those specific turns. And then more. And then more. Lots of models, lots of retraining, an endless loop of madness! I sometimes fell into the trance of the Strange Loop of the track. At 2AM the Udacity workspace froze and my data in `/opt` was gone forever. Lots of data, retraining, ...  Somehow, the car would not collide into rectangular barriers. Eventually, it would actually turn _while in the center of the road_. Amazing. I've finally learned how to drive better than my desired CNN! And so I became one with the machine. Is it a dream? Am I inside this world now? Lots of data, retraining, ...  I digress.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. _clap_ _clap_

#### 2. Final Model Architecture

The final model architecture consisted of a CNN with five convolutional layers and three fully-connected layers, utilizing ReLU activation layers to introduce nonlinearity. It has been described earlier.

Here is a visualization of the architecture. Details of the architecture could be seen here (some parts were modified in my model, most notably the presence of dropout layers; refer to code in `model.py`).

![alt text][nvidia]

#### 3. Creation of the Training Set & Training Process

My first lap, I had difficulty staying on the road and went over some lane lines, and later laps had slightly better training data. Afterwards, to deal with vehicle oscillations, I focused much more on the recovery from left + right sides of the road. Next, noticing the model performed glaringly poorly in some turns, I focused in collecting data for those turns, using a separate data directory to keep epoch-training times lower.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back towards center when it is off to the sides. These images show what a recovery looks like starting from being oriented towards the edge of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also horizontally flipped images and angles. Since I went counter-clockwise around the track for my training data, this flipping would replicate going clockwise around the track.

After the collection process, I had 25k+ data points. Data augmenting via horizontal-flipping doubled this amount. I then preprocessed this data by cropping out the top and bottom portion of each image, leaving only the salient portion of the road.

I finally randomly shuffled the data set and placed 10% of the data into a validation set, formed using sklearn.model_selection.train_test_split.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an Adam optimizer so that manually tuning the learning rate wasn't necessary.

__Appendix on Training difficulties (ranting)__

* Multiple data sets with varying amounts of center-road driving and responding to particular turns (car would often crash into the diamond-shaped corners of the bridge or the brown road after -- it seemed very attracted to corners). I took a _lot_ of data to correct turns at these places.
* I also tried cropping the images to various amounts and even resizing (to speed up training and avoid GPU Out of Memory errors -- my model performed poorly with these resizes and I found "Refresh Workspace" resolved GPU OOM, especially if I was testing the model concurrently with performing training, e.g. with that model file loaded).
* I was saving my data in /opt/carnd_p3/turns_data and one time the machine froze and everything in /opt/carnd_p3 was reset, losing all my data
* I created multiple models of varying architectures and levels of training of each of the dataset, to compare driving behavior. Finally, I arrived at one which was smooth and did not attract itself to edges. Data collecting took ~10 hours in total, and this was entirely aimed to get the car to complete a loop around the first track.
* The car would frequently swerve in straight sections (especially the early one) at my desired 30mph. So I recorded the video at an agonizingly 10mph so the vehicle could respond fast enough and avert the edge-attracting disasters that has long plagued these 15~ GPU workspace hours.
* At one point, I gave up collecting my own data (I had difficulty keeping to center of road and steering well myself). I got my dad to gather some data.
