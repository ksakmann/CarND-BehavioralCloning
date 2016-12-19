import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.optimizers import Adam
from keras.models import model_from_json
import json

## model input files for initial model save and retrain
fileDataPath = './data'
#fileDataPath = './FirstTrackDa'
fileDataCSV = '/driving_log.csv'
fileModelJSON = 'model1.json'
fileWeights = 'model1.h5'

## model parameters defined here
#batch_size = 512
#nb_epoch = 10
#img_rows, img_cols = 160, 320
#nb_classes = 1


training_dat=pd.read_csv(fileDataPath+fileDataCSV)
training_dat.columns = ['center_image', 'left_image','right_image','steering_angle','throttle','break','speed']
df = pd.DataFrame({'center_image': training_dat['center_image'],'steering_angle': training_dat['steering_angle']},
                  columns=['center_image','steering_angle'])

X_train = np.copy(df['center_image'])
Y_train = np.copy(df['steering_angle'])

print(df.head())

Y_train = Y_train.astype(np.float32)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

batch_size = 2
samples_per_epoch = len(X_train)/batch_size
val_size = int(samples_per_epoch/10.0)
nb_epoch = 1000

# input image dimensions
img_rows, img_cols, img_ch = 160, 320, 3
print('image shape :',img_rows,img_cols,img_ch)
print('image data type :',X_train.dtype)
print('label data type :',Y_train.dtype)


def load_image(imagepath):
    imagepath = fileDataPath+'/'+imagepath
    imagepath = imagepath.replace(' ','')
    image = cv2.imread(imagepath, 1)    
    shape = image.shape    
    image = image[int(shape[0]/3):shape[0], 0:shape[1]]
    image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)    
    return image

def batchgen(X, Y):
    while 1:
        for i in range(len(X)):            
            imagepath,y = X[i],Y[i]
            #p=np.random.rand(1)
            #if abs(y)<0.03 and p>0.75: 
            #    j = np.random.randint(0,len(X))
            #    imagepath,y = X[j],Y[j]            
            #print("imagepath: ", '"'+imagepath+'"', "steering: ", y)
            image = load_image(imagepath)
            coin=np.random.rand(1)
            if coin > 0.5:
                image,y=cv2.flip(image,1),-y
            # print("image: ", image.shape, "steering: ", y)
            y = np.array([[y]]).astype(np.float32)
            image = image.reshape(1, img_rows, img_cols, img_ch)
            # print("image: ", image.shape, "y: ", y.shape)
            yield image, y


# Model Building 
# Start with a self-made model

# number of convolutional filters to use
nb_filter1 = 32
nb_filter2 = 64
nb_filter3 = 128
# size of pooling area for max pooling
pool_size = (2, 2)
pool_strides = (1,1)
# convolution kernel size
kernel_size = (5, 5)
# number of hidden units in the first fully connected layer
nb_fc1=128
nb_fc2=128

model = Sequential()
model.add(Lambda(lambda x: x/255.0,input_shape=(img_rows, img_cols,img_ch),output_shape=(img_rows, img_cols,img_ch)))
model.add(Convolution2D(nb_filter1, kernel_size[0], kernel_size[1],border_mode='same', subsample=(2,2),name='conv1'))
model.add(Activation('relu',name='relu1'))
model.add(MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='maxpool1'))
model.add(Convolution2D(nb_filter2, kernel_size[0], kernel_size[1],border_mode='same',subsample=(2,2),
                        name='conv2'))
model.add(Activation('relu',name='relu2'))
model.add(MaxPooling2D(pool_size=pool_size,strides=None,name='maxpool2'))
model.add(Convolution2D(nb_filter2, kernel_size[0], kernel_size[1],border_mode='same',subsample=(1,1),
                        name='conv3'))
model.add(Activation('relu',name='relu3'))
model.add(MaxPooling2D(pool_size=pool_size,strides=None,name='maxpool3'))
model.add(Flatten(name='flatten'))
model.add(Dropout(0.5,name='dropout1'))
model.add(Dense(nb_fc1, name='hidden1'))
model.add(Activation('relu',name='relu4'))
model.add(Dropout(0.5,name='dropout2'))
model.add(Dense(nb_fc2,  name='hidden2'))
#model.add(Activation('relu',name='relu5'))
model.add(Dense(1, name='output'))
model.summary()


adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

restart=True
if os.path.isfile(fileModelJSON) and restart:
    try:
        with open(fileModelJSON) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(fileWeights)    
        print('loading trained model ...')
    except Exception as e:
        print('Unable to load model', model_name, ':', e)
        raise    

model.compile(optimizer=adam, loss='mean_squared_error')

history = model.fit_generator(batchgen(X_train, Y_train),
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                    validation_data=batchgen(X_val, Y_val),
                    nb_val_samples=val_size,
                    verbose=1)


json_string = model.to_json()
with open(fileModelJSON, 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights(fileWeights)


# test a few predictions
#n_preds=20
#y_preds=model.predict(X_train[:n_preds],verbose=1)
#print(y_preds,y_train[:n_preds])
