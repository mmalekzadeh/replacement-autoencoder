#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September 2017

@author: mmalekzadeh
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import model_from_json
#####################################
# fix random seed for reproducibility
np.random.seed(2017)
#####################################
batch_size = 32
num_of_epochs = 2

w_train_data = np.load("data_train_white.npy")
w_test_data = np.load("data_test_white.npy")
b_train_data = np.load("data_train_black.npy")
b_test_data = np.load("data_test_black.npy") 
g_train_data = np.load("data_train_gray.npy")
g_test_data = np.load("data_test_gray.npy") 

rnd_idx_train = np.random.choice(g_train_data.shape[0], b_train_data.shape[0], replace=False)
rnd_idx_test = np.random.choice(g_test_data.shape[0], b_test_data.shape[0], replace=False)

b_train_transformed = g_train_data[rnd_idx_train,:]
b_test_transformed = g_test_data[rnd_idx_test,:]

x_train = np.append(w_train_data, g_train_data, axis=0)
x_train = np.append(x_train, b_train_data, axis=0)
x_test = np.append(w_test_data, g_test_data, axis=0)
x_test = np.append(x_test, b_test_data, axis=0)
x_train_transformed = np.append(w_train_data, g_train_data, axis=0)
x_train_transformed = np.append(x_train_transformed , b_train_transformed, axis=0)
x_test_transformed = np.append(w_test_data, g_test_data, axis=0)
x_test_transformed = np.append(x_test_transformed , b_test_transformed, axis=0)

resh=np.prod(w_train_data.shape[1:])
x_train = x_train.reshape((len(x_train), resh))
x_test = x_test.reshape((len(x_test), resh))
x_train_transformed = x_train_transformed.reshape((len(x_train_transformed), resh))
x_test_transformed = x_test_transformed.reshape((len(x_test_transformed), resh))

##### Replacement Autoencoder #######
input_img = Input(shape=(resh,))

x = Dense(resh, activation='linear')(input_img)

encoded = Dense(resh//2, activation='selu')(x)
encoded = Dense(resh//8, activation='selu')(encoded)

y = Dense(resh//16, activation='selu')(encoded)

decoded = Dense(resh//8, activation='selu')(y)
decoded = Dense(resh//2, activation='selu')(decoded)

z = Dense(resh, activation='linear')(decoded)

autoencoder = Model(input_img, z)

autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train , x_train_transformed,
                epochs= num_of_epochs,
                batch_size = batch_size,
                shuffle=True,
                validation_data=(x_test, x_test_transformed)
                )

# serialize model to JSON
model_json = autoencoder.to_json()
with open("rae_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

autoencoder.save_weights("rae_model_weights.h5")
print("Saved model to disk")


rw_test_data = w_test_data.reshape((len(w_test_data), resh))
rb_test_data = b_test_data.reshape((len(b_test_data), resh))
rg_test_data = g_test_data.reshape((len(g_test_data), resh))

transformed_w_test_data = autoencoder.predict(rw_test_data)
transformed_b_test_data = autoencoder.predict(rb_test_data)
transformed_g_test_data = autoencoder.predict(rg_test_data)

transformed_w_test_data = transformed_w_test_data.reshape((len(w_test_data), w_test_data.shape[1], w_test_data.shape[2], 1))
transformed_b_test_data = transformed_b_test_data.reshape((len(b_test_data), b_test_data.shape[1], b_test_data.shape[2], 1))
transformed_g_test_data = transformed_g_test_data.reshape((len(g_test_data), g_test_data.shape[1], g_test_data.shape[2], 1))

### Saving Datasets ###
np.save("transformed_w_test_data.npy", transformed_w_test_data)
np.save("transformed_b_test_data.npy", transformed_b_test_data)
np.save("transformed_g_test_data.npy", transformed_g_test_data)
print("Info: Transformed data are saved")
