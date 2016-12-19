# CarND-BehavioralCloning
In this project for the Udacity Self-Driving Car Nanodegree a deep CNN  is developed that can steer a car in a simulator provided by Udacity. 
The car drives autonomously around the track. The network was trained on the images from a video stream that was recorded while 
a human was driving the car. The CNN thus clones the human driving behavior.

# General considerations.
The training track is a round track with  sharp corners, exits and entries, bridges, partially missing lane lines 
and changing light conditions. An additional test track exists with changing elevations, even sharper turns and bumps. 
It is thus crucial that the CNN does not merely memorizes the first track, but generalizes to unseen data. 

# Model architecture
CNNs architectures have been successfully used to predict the steering angle of the simulator. 
Among these are the CNN architecture of NVIDIA https://arxiv.org/pdf/1604.07316v1.pdf or the comma.ai architecture 
https://github.com/commaai/research

In both cases a single variable is predicted, namely the current steering angle as a real valued number. The problem is thus not 
a classification but rather of the regression type (it would be interesting though to see how a discretized version performs).

In https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.4iywd3mzj the author Vivek Yadav provided a solution 
to the steering prediction problem based on data the judicious use of data augmentation. We follow a very similar approach here,
but differing in the network architecture and augmentation techniques. 

For the network architecture we chose a pretty standard convolutional architecture (essentially 
the same as used earlier for classfying traffic signs https://github.com/ksakmann/CarND-TrafficSignClassifier)
but eliminating pooling operations. These tend to make the outcome translationally shift invariant, 
which is desirable for classification tasks, but not for  keeping a car in the middle of the road.




