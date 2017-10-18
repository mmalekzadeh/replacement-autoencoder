#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September 2017

@author: malekzadeh
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.models import model_from_json
import numpy as np
from PIL import Image
import math
#### Global Variables
o_gray = np.load("data_test_gray.npy")
tr_gray = np.load("transformed_g_test_data.npy")
tr_black = np.load("transformed_b_test_data.npy")

num_of_sensors = o_gray.shape[1]
itr = 0    
batch_szie = 64
iteration = 101
resultes = np.zeros((4,iteration))

def calcul():
    global resultes
    global itr
    global tr_gray
    global tr_black

    # load json and create model
    d_json_file = open('dmh.json', 'r')
    d_json = d_json_file.read()
    d_json_file.close()
    dm = model_from_json(d_json)
    # load weights into new model
    dm.load_weights("dwh.h5")
    print("Loaded D from disk")

    # load json and create model
    g_json_file = open('gmh.json', 'r')
    g_json = g_json_file.read()
    g_json_file.close()
    gm = model_from_json(g_json)
    # load weights into new model
    gm.load_weights("gwh.h5")
    print("Loaded G from disk")
    
    dm.compile(loss='binary_crossentropy',
                     optimizer='SGD',
                     metrics=['accuracy'])
    gm.compile(loss='binary_crossentropy',
                     optimizer='SGD',
                     metrics=['accuracy'])

    #Random
    noise = np.random.normal(0, 1, size=(1000, 100))
    gmgray1 = gm.predict(noise, verbose=0)

    #Best 10%
    numo_of_samples=10000    
    noise = np.random.normal(0, 1, (numo_of_samples, 100))
    generated_images = gm.predict(noise, verbose=1)
    d_pret = dm.predict(generated_images, verbose=1)
    index = np.arange(0, numo_of_samples)
    index.resize((numo_of_samples, 1))
    pre_with_index = list(np.append(d_pret, index, axis=1))
    pre_with_index.sort(key=lambda x: x[0], reverse=True)
    nice_images = np.zeros((1000,) + generated_images.shape[1:3], dtype=np.float32)
    nice_images = nice_images[:, :, :, None]
    for i in range(1000):
        idx = int(pre_with_index[i][1])
        nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
    # Creat labels for each kind of data
    yg = [1]*tr_gray.shape[0]
    yb = [0]*tr_black.shape[0]
    ygm1 = [0]*1000
    ygm10 = [0]*1000
    resultes[0,itr] = (dm.evaluate(tr_gray,yg,verbose=1))[1]
    resultes[1,itr] = (dm.evaluate(tr_black,yb,verbose=1))[1]
    resultes[2,itr] = (dm.evaluate(gmgray1,ygm1,verbose=1))[1]
    resultes[3,itr] = (dm.evaluate(nice_images,ygm10,verbose=1))[1]
    print("\n Accuracy on Original Gray : %s: %.2f%%" % (dm.metrics_names[1], resultes[0,itr]*100))
    print("\n Accuracy on Fake Gray(Black) : %s: %.2f%%" % (dm.metrics_names[1], resultes[1,itr]*100))
    print("\n Accuracy on Generated Gray : %s: %.2f%%" % (dm.metrics_names[1], resultes[2,itr]*100))
    print("\n Accuracy on top 10 Generated Gray : %s: %.2f%%" % (dm.metrics_names[1], resultes[3,itr]*100))
    itr = itr+1;
    np.savetxt('GAN_Results.txt', resultes)

    
def generator_model():
    global num_of_sensors
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('selu'))
    model.add(Dense(128*num_of_sensors*5))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Reshape((num_of_sensors, 5, 128), input_shape=(128*num_of_sensors*5,)))
    model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, (1, 5), padding='same'))
    model.add(Activation('selu'))
    model.add(UpSampling2D(size=(1, 3)))
    model.add(Conv2D(1, (1, 5), padding='same'))
    model.add(Activation('linear'))
    return model


def discriminator_model():
    global num_of_sensors
    model = Sequential()
    model.add(
            Conv2D(64, (1, 5),
            padding='same',
            input_shape=(num_of_sensors, 30, 1))
            )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(64, (1, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Conv2D(64, (1, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    global o_gray
    global iteration
    X_train = o_gray
    X_train = X_train[::8]
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2]))
    X_train = X_train[:, :, :, None]
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(iteration):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
        if epoch % 5 == 0:
            print("batch %d d_loss : %f" % (index, d_loss))
            print("batch %d g_loss : %f" % (index, g_loss))
            d_json = d.to_json()
            with open("dmh.json", "w") as d_json_file:
                d_json_file.write(d_json)
            d.save_weights("dwh.h5")
            print("Saved D to disk")
            
            g_json = g.to_json()
            with open("gmh.json", "w") as g_json_file:
                g_json_file.write(g_json)
            g.save_weights("gwh.h5")
            print("Saved G to disk")
            calcul()

#### Call GAN
train(BATCH_SIZE = batch_szie)
