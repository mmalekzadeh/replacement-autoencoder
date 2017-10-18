#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created  September  2017
    To evaluate the performance of time-series transformation by RAE,
    This modul implements a ConvNet as a third party model (server model).
    
@author: mmalekzadeh
"""
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils 

#####################################
# fix random seed for reproducibility
np.random.seed(2017)
#####################################

###### Global Variables ###
train_data = np.load("sections_data_train.npy")
train_label = np.load("sections_label_train.npy")
test_data = np.load("sections_data_test.npy")
test_label = np.load("sections_label_test.npy")

## in each iteration, we consider 'batch_size' training examples at once
batch_size = 128 
## we iterate 'num_epochs' times over the entire training set   
num_epochs = 10
## Size of CNN kernels
kernel_size_1 = 5
kernel_size_2 = 3
## Size of Pooling operators
pool_size_1 = 2
pool_size_2 = 3  
## Size of CNN layers
conv_depth_1 = 50 
conv_depth_2 = 40 
conv_depth_3 = 20 
## To dropout after pooling layers
drop_prob_1 = 0.4 
## To dropout in the Flatten layers
drop_prob_2 = 0.6 
## Size of the Last hidden layer before Output layer
hidden_size = 400 
num_classes = np.unique(train_label).shape[0] 
### One-hot encode the labels
y_train = np_utils.to_categorical(train_label, num_classes) 
y_test = np_utils.to_categorical(test_label, num_classes) 
num_train, height, width = train_data.shape
num_test = test_data.shape[0] 

#### Building CNN
## Input Layer
inp = Input(shape=(height, width,1)) 
## Convnet -- Layer 1 and 2 
conv_1 = Convolution2D(conv_depth_1, (1 , kernel_size_1),
                       padding='valid', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (1 , kernel_size_2),
                       padding='same', activation='relu')(conv_1)
dense_2 = Dense(conv_depth_1, activation='relu')(conv_2)
pool_2 = MaxPooling2D(pool_size=(1, pool_size_1))(dense_2)
drop_2 = Dropout(drop_prob_1)(pool_2)
## Convnet -- Layer 3 
conv_3 = Convolution2D(conv_depth_2, (1 , kernel_size_1),
                       padding='valid', activation='relu')(drop_2)
dense_3 = Dense(conv_depth_2, activation='relu')(conv_3)
pool_3 = MaxPooling2D(pool_size=(1, pool_size_2))(dense_3)
drop_3 = Dropout(drop_prob_1)(pool_3)
## Convnet -- Layer 4 
conv_4 = Convolution2D(conv_depth_3, (1 , kernel_size_2),
                       padding='valid', activation='relu')(drop_3)
drop_4 = Dropout(drop_prob_1)(conv_4)
## Flatten Layer
flat = Flatten()(drop_4)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_5 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_5)
# To define a model , we specify its input and output layers
model = Model(inputs=inp, outputs=out) 
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

# Train the model using the training set                       
train_data = np.expand_dims(train_data,axis=3)
test_data = np.expand_dims(test_data,axis=3)
model.fit(train_data, y_train,                
          batch_size = batch_size,
          epochs = num_epochs,
          verbose = 1,
          )
# Evaluate the trained model on the test set!          
scores = model.evaluate(test_data, y_test, verbose=1)
print("\n ~~~ Result: %s of server model: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#serialize model to JSON
model_json = model.to_json()
with open("server_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("server_cnn_model_weights.h5")
print("Info: Saved model and weights to disk")
