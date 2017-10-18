#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 2017

This module builds a new dataset from training and testing timseries dataset.
It selects a size for its sliding window and iteratively take a section from
original data and store it in a new three-dimensional matrix.

@author: mmalekzadeh
"""
import numpy as np
import sys

### Global Variables ###
train_data = np.load("clean_training_dataset.npy")
test_data = np.load("clean_testing_dataset.npy")

##  This Variable Defines the Size of Sliding Window.
##  ( e.g. 30 means in each section we just consider 30...
##    consecutive observations of each sensor) 
sliding_window_size = 30 
##  Here We Choose Step Size for Building Diffrent sections from Time-Series.
##  ( smaller step size will increase the amount of the instances... 
##    and higher computational cost may be incurred )
step_size_of_sliding_window = 3

def build_sections():
    global train_data
    global test_data    
    ##  The Last Column in the Dataset Contains label of Classes 
    size_features = train_data.shape[1] - 1
    size_train = train_data.shape[0]
    size_test = test_data.shape[0]    
    ##  Create a 3D matrix for Storing Sections  
    sections_train = np.zeros((size_train , size_features , sliding_window_size ))
    sections_test = np.zeros((size_test, size_features , sliding_window_size ))    
    labels_train = np.zeros(size_train)
    labels_test = np.zeros(size_test)
    # for randomly removing some Null samples : hold "(100/rnd)%" of Null samples
    rnd = 5 
    # Training Data
    nulls=0
    jtr=0
    for i in range(0 ,(size_train) - sliding_window_size , step_size_of_sliding_window): 
        lbl = np.argmax(np.bincount(train_data[i:i+sliding_window_size, -1].astype(int)))                      
        if lbl==0 and nulls%rnd!=0:
            nulls+=1    
        else:
            if lbl==0:
                nulls+=1  
            one_sec = (train_data[i:i+sliding_window_size, 0:size_features]).T
            sections_train[jtr] = one_sec
            labels_train[jtr] = lbl
            jtr = jtr+1
            sys.stdout.write("\r {} from {}".format(i,(size_train) - sliding_window_size ))
            sys.stdout.flush() 
    sections_train = sections_train[0:jtr, :]
    labels_train = labels_train[0:jtr]
    print("\nInfo: Size of Training Sections = {}".format(sections_train.shape))

    # Testing Data 
    nulls=0
    jte=0
    for i in range(0 ,(size_test) - sliding_window_size , step_size_of_sliding_window):
        lbl = np.argmax(np.bincount(test_data[i:i+sliding_window_size,-1].astype(int)))                      
        if lbl==0 and nulls%rnd!=0:
            nulls+=1    
        else:
            if lbl==0:
                nulls+=1  
            one_shot = (test_data[i:i+sliding_window_size,0:size_features]).T
            sections_test[jte] = one_shot
            labels_test[jte] = lbl
            jte = jte+1
            sys.stdout.write("\r {} from {}".format(i,(size_test) - sliding_window_size ))
            sys.stdout.flush()
    sections_test = sections_test[0:jte, :] 
    labels_test = labels_test[0:jte]       
    print("\nInfo: Size of Testing Sections = {}".format(sections_test.shape))
    
    ### Saving Datasets ###
    np.save("sections_data_train.npy", sections_train)
    np.save("sections_label_train.npy", labels_train)
    print("Info: Training Sections are saved")
    np.save("sections_data_test.npy", sections_test)
    np.save("sections_label_test.npy", labels_test)
    print("Info: Testing Sections are saved")
    
    return

### Calling desired Moduls 
"""
    1) "build_sections" steps on time-series data and extract temporal sections.
"""
build_sections()