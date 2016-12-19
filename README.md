# CarND-BehavioralCloning
In this project for the Udacity Self-Driving Car Nanodegree a deep CNN  is developed that can steer a car in a simulator provided by Udacity. The car drives autonomously around the track. The network was trained on the images from a video stream that was recorded while a human was driving the car. The CNN thus clones the human driving behavior.

# General considerations.
The simulated car is equipped with three cameras on the left center and right of the driver that provide images from these view points. The training track is a round track with  sharp corners, exits and entries, bridges, partially missing lane lines 
and changing light conditions. An additional test track exists with changing elevations, even sharper turns and bumps. 
It is thus crucial that the CNN does not merely memorizes the first track, but generalizes to unseen data, in order to perform well also on the test track. The model provided here was trained exclusively on the first track and completes the test track.
The main problem lies in the skew of the data set: most of the time the steering angle is small or zero, but the important events are when the car needs to turn sharply. The correct steering angles are not easy to generate by manually approaching the side of the road, because still most of the time the car drives straight then, with the exception of the short moment when the driver avoids a crash or the car going off the road. We Therefore decided to train the car by driving as smothly as possible right in the middle of the road and simulating all recovery events by transforming the images obtained this way with corresponding steering angle changes.

# Model architecture
CNNs architectures have been successfully used to predict the steering angle of the simulator. 
Among these are the CNN architecture of NVIDIA https://arxiv.org/pdf/1604.07316v1.pdf or the comma.ai architecture 
https://github.com/commaai/research which were used successfully, e.g. in the submission of https://github.com/diyjac/SDC-P3.

In both architectures a single variable is predicted, the current steering angle as a real valued number. The problem is thus not a classification but a regression task and we will follow the same approach (it would be interesting though to see how a discretized version performs).

In https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.4iywd3mzj the author provided a solution 
to this steering problem based on the judicious use of data augmentation. This submission draws on the insights obtained there 
but differs in the network architecture and augmentation techniques. 

For the network architecture we use a CNN that evolved from a previous submiossion for classfying traffic signs https://github.com/ksakmann/CarND-TrafficSignClassifier but with some changes. 
The network starts with a preprocessing layer that takes in images of shape 64x64x3. The image gets normalized to the range [-1,1] otherwise no preprocessing is performed. Following the input layer are 4 convolutional layers with ReLU activations. The first two layers employ kernels of size k=(8,8) with a stride of s=(4,4) and 32 and 64 channels, respectively. The next convolutional layer uses k=(4,4) kernels, a stride of s=(2,2) and 128 channels. In the last convolutional layer we use k=(2,2), a stride s=(1,1) and again 128 channels. Following the convolutional layers are two fully connected layers  with ReLU activations as well as dropout regularization right before the layers. The final layer is a single neuron that provides the predicted steering angle. We explicitly avoided the use of pooling layers because pooling layers apart from down sampling also provide (some) shift invariance, which is desirable for classification tasks, but is counterproductive for keeping a car centered on the road (note: the comma.ai architecture does not use pooling either).

The training images are generated on the fly from a data set gathered few rounds driving around the track in one direction.
During training a python generator creates new images with accordingly corrected steering angles. The operations performed  are 

0. A random training example is chosen
1. The camera (left,right,center) is chosen randomly
2. Random shear: the image is sheared horizontally to simulate a bending road
3. Random crop: we randomly crop a frame out of the image to simulate the car being offset from the middle of the road (also downsampling the image to 64x64x3 is done in this step)
4. Random flip: to make sure left and right turns occur just as frequently 
5. Random brightness: to simulate differnt lighting conditions

In steps 1-4 the steering angle is adjusted to account for the change of the image.
