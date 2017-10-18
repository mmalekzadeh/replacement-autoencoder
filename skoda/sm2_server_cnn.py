#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created  September  2017

@author: mmalekzadeh
"""
import numpy as np
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import model_from_json
#####################################
# fix random seed for reproducibility
np.random.seed(2017)
#####################################

#### Global Variables ###
train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")
train_label = np.load("train_labels.npy")
test_label = np.load("test_labels.npy")

batch_size = 64
num_of_epochs = 2
kernel_size_1 = 5
kernel_size_2 = 3

pool_size_1 = 2
pool_size_2 = 3  

conv_depth_1 = 50 
conv_depth_2 = 40 
conv_depth_3 = 20 

drop_prob_1 = 0.4 

drop_prob_2 = 0.6 

hidden_size = 400 

num_classes = np.unique(train_label).shape[0] 
y_train = np_utils.to_categorical(train_label, num_classes) 
y_test = np_utils.to_categorical(test_label, num_classes) 

num_train, height, width = train_data.shape
num_test = test_data.shape[0] 

inp = Input(shape=(height, width,1)) 

conv_0 = Convolution2D(conv_depth_1, (1 , kernel_size_1), padding='valid', activation='relu')(inp)
conv_1 = Convolution2D(conv_depth_1, (1 , kernel_size_2), padding='same', activation='relu')(conv_0)
dense_1 = Dense(conv_depth_1, activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(1, pool_size_1))(dense_1)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_2 = Convolution2D(conv_depth_2, (1 , kernel_size_1), padding='valid', activation='relu')(drop_1)
dense_2 = Dense(conv_depth_2, activation='relu')(conv_2)
pool_2 = MaxPooling2D(pool_size=(1, pool_size_2))(dense_2)
drop_2 = Dropout(drop_prob_1)(pool_2)

conv_3 = Convolution2D(conv_depth_3, (1 , kernel_size_2), padding='valid', activation='relu')(drop_2)
drop_3 = Dropout(drop_prob_1)(conv_3)

flat = Flatten()(drop_3)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_4 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_4)

model = Model(inputs=inp, outputs=out) 

train_data = np.expand_dims(train_data,axis=3)
test_data = np.expand_dims(test_data,axis=3)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

                       
model.fit(train_data, y_train,                
          batch_size = batch_size,
          epochs = num_of_epochs,
          verbose=1,
          ) 
          
scores = model.evaluate(test_data, y_test, verbose=1)  # Evaluate the trained model on the test set!
print("\n ~~~ Result: \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#serialize model to JSON
model_json = model.to_json()
with open("server_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("server_cnn_model_weights.h5")
print("Saved model to disk")
