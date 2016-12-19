import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.optimizers import Adam
from keras.models import model_from_json
import json

## model input files for initial model save and retrain
fileDataPath = './data'
fileDataCSV = '/driving_log.csv'
fileModelJSON = 'model0.json'
fileWeights = 'model0.h5'

## model parameters defined here
#batch_size = 512
#nb_epoch = 10
#img_rows, img_cols = 160, 320
#nb_classes = 1


matplotlib.style.use('ggplot')
training_dat=pd.read_csv(fileDataPath+fileDataCSV)
training_dat.columns = ['center_image', 'left_image','right_image','steering_angle','throttle','break','speed']

df = pd.DataFrame({'center_image': training_dat['center_image'],'steering_angle': training_dat['steering_angle']},
                  columns=['center_image','steering_angle'])

#plt.figure();
#df.plot.hist(bins=41,alpha=0.5)

#df1 = df[abs(df['steering_angle'])>0.001]
#df2 = df[abs(df['steering_angle'])<0.001]
#df3 = df2.iloc[::10, :]
#frames=[df1,df3]
#df_train=pd.concat(frames)
#df_train.plot.hist(bins=nb_classes,alpha=0.5)

# Read in training data 
X_train=[]
y_train=[]

for index,row in df.iterrows():
    x,y=(plt.imread(fileDataPath+'/'+df['center_image'][index])).astype(np.float32),(df['steering_angle'][index]).astype(np.float32)
    X_train.append(x)
    y_train.append(y)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

# bin and map the steering angles to nb_classes classes
nb_classes=41
y_digitized=np.round((y_train)/2.0*(nb_classes-1))
y_digitized=y_digitized.astype(np.int32)
np.unique(y_digitized)
y_train = y_digitized.astype(np.float32)/(nb_classes//2)


# split training set in validation and a new training set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)


# Model Building 
# Start with a self-made model

# input image dimensions
shape=shape=X_train.shape[1:]
img_rows, img_cols, img_ch = shape
print('image shape :',shape)
print('image data type :',X_train.dtype)
print('label data type :',y_train.dtype)

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

nb_epoch = 100
batch_size=128

adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

if os.path.isfile(fileModelJSON):
    try:
        with open(fileModelJSON) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(fileWeights)    
        print('loading trained model ...')
    except Exception as e:
        print('Unable to load model', model_name, ':', e)
        raise    

model.compile(optimizer=adam, loss='mse',metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    validation_data=(X_val, y_val)
                    verbose=1)



json_string = model.to_json()
with open(fileModelJSON, 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights(fileWeights)


# test a few predictions
n_preds=20
y_preds=model.predict(X_train[:n_preds],verbose=1)
print(y_preds,y_train[:n_preds])
