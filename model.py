import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.optimizers import Adam
from keras.models import model_from_json
import json

matplotlib.style.use('ggplot')

data_dir = './data/mydata'
data_csv = '/driving_log.csv'
model_json = 'model.json'
model_weights = 'model.h5'

#col_names = ['center', 'left','right','steering','throttle','brake','speed']
training_dat = pd.read_csv(data_dir+data_csv,names=None)
training_dat.head()


training_dat[['left','center','right']]
X_train = training_dat[['left','center','right']]
Y_train = training_dat['steering']

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# get rid of the pandas index after shuffling
X_left  = X_train['left'].as_matrix()
X_right = X_train['right'].as_matrix()
X_train = X_train['center'].as_matrix()
X_val   = X_val['center'].as_matrix()
Y_val   = Y_val.as_matrix()
Y_train = Y_train.as_matrix()

Y_train = Y_train.astype(np.float32)
Y_val   = Y_val.astype(np.float32)


if data_dir=='./data/udacity':
    X_train = X_train.apply(lambda x: data_dir+'/'+x)
    X_left = X_left.apply(lambda x: data_dir+'/'+x)
    X_right = X_right.apply(lambda x: data_dir+'/'+x)

print('X_train[:5]: ',X_train[:5])
print('Y_train[:5]: ',Y_train[:5])


def read_next_image(m,lcr,X_train,X_left,X_right,Y_train):
    # assume the side cameras are about 1.2 meters off the center and the offset to the left or right 
    # should be be corrected over the next dist meters, calculate the change in steering control
    # using tan(alpha)=alpha

    offset=1.0 
    dist=20.0
    steering = Y_train[m]
    if lcr == 0:
        image = plt.imread(X_left[m].strip(' '))
        dsteering = offset/dist * 360/( 2*np.pi) / 25.0
        steering += dsteering
    elif lcr == 1:
        image = plt.imread(X_train[m].strip(' '))
    elif lcr == 2:
        image = plt.imread(X_right[m].strip(' '))
        dsteering = -offset/dist * 360/( 2*np.pi)  / 25.0
        steering += dsteering
    else:
        print ('Invalid lcr value :',lcr )
    
    return image,steering

def random_crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0
    
    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift 
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/3.0
    else:
        dsteering = 0
    steering += dsteering
    
    return image,steering

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    
    return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering
        

def generate_training_example(X_train,X_left,X_right,Y_train):
    m = np.random.randint(0,len(Y_train))
#    print('training example m :',m)
    lcr = np.random.randint(0,3)
    #lcr = 1
#    print('left_center_right  :',lcr)
    image,steering = read_next_image(m,lcr,X_train,X_left,X_right,Y_train)
#    print('steering :',steering)
#    plt.imshow(image)
    image,steering = random_shear(image,steering,shear_range=100)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)    
    image,steering = random_crop(image,steering,tx_lower=-20,tx_upper=20,ty_lower=-10,ty_upper=10)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)
    image,steering = random_flip(image,steering)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)
    
    image = random_brightness(image)
#    plt.figure()
#    plt.imshow(image)
    
    return image,steering

def get_validation_set(X_val,Y_val):
    X = np.zeros((len(X_val),64,64,3))
    Y = np.zeros(len(X_val))
    for i in range(len(X_val)):
        x,y = read_next_image(i,1,X_val,X_val,X_val,Y_val)
        X[i],Y[i] = random_crop(x,y,tx_lower=0,tx_upper=0,ty_lower=0,ty_upper=0)
    return X,Y
    

def generate_train_batch(X_train,X_left,X_right,Y_train,batch_size = 32):
    
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = generate_training_example(X_train,X_left,X_right,Y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering



batch_size=200
train_generator = generate_train_batch(X_train,X_left,X_right,Y_train,batch_size)
X_val,Y_val = get_validation_set(X_val,Y_val)

print('X_train data type :',X_train.dtype)
print('Y_train data type :',Y_train.dtype)
print('X_val data type :',X_val.dtype)
print('Y_val data type :',Y_val.dtype)


model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
model.add(Activation('relu',name='relu2'))
model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1))
model.summary()


adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

restart=True
if os.path.isfile(model_json) and restart:
    try:
        with open(model_json) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(model_weights)    
        print('loading trained model ...')
    except Exception as e:
        print('Unable to load model', model_name, ':', e)
        raise    

model.compile(optimizer=adam, loss='mse')


nb_epoch=10
history = model.fit_generator(train_generator,
                    samples_per_epoch=20000, nb_epoch=nb_epoch,
                    validation_data=(X_val,Y_val),verbose=1)

json_string = model.to_json()

print('Save the model')

try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass   

with open(model_json, 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights(model_weights)

print('Done')
