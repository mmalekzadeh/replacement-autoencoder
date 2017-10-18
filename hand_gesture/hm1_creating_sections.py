#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created September 2017

@author: mmalekzadeh
"""
import numpy as np
from scipy import stats

#####################################
# fix random seed for reproducibility
np.random.seed(2017)
#####################################

def create_sections(subject):
    if(subject == 1):
        data = np.loadtxt("subject_1_data.txt",delimiter=',')
        labels = np.loadtxt("subject_1_labels.txt",delimiter=',')
    else:    
        data = np.loadtxt("subject_2_data.txt",delimiter=',')
        labels = np.loadtxt("subject_2_labels.txt",delimiter=',')

    # ## normalize each sensor’s data to have a zero mean and unity standard deviation.
    data = stats.zscore(data, axis=0, ddof=0)

    data = data.T
    labels = labels.T

    ##  This Variable Defines the Size of Sliding Window
    ##  ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor) 
    sliding_window_size = 30
    
    ##  Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
    ##  ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
    step_size_of_sliding_window = 3
        
    size_features = data.shape[0]
    size_data = data.shape[1]
    
        
    number_of_sections = round(((size_data - sliding_window_size)/step_size_of_sliding_window))
        
    ##  Create a 3D matrix for Storing Snapshots  
    all_data_shots= np.zeros((number_of_sections , size_features , sliding_window_size ))
    all_labels_shots = np.zeros(number_of_sections)
    
    for i in range(0 ,(size_data) - sliding_window_size , step_size_of_sliding_window):
        all_data_shots[i // step_size_of_sliding_window ] = data[0:size_features, i:i+sliding_window_size]
        all_labels_shots[i // step_size_of_sliding_window] = np.argmax(np.bincount(labels[i:i+sliding_window_size].astype(int)))
        
    """
    1) above code steps on time-series data and extract temporal snapshots.
    We build this new kind of dataset because its more flexible for Convolutional Neural Networks
    """
    
    ## There are 11 Gestures classes in dataset
    ## Respectively :
        ##    0  'NULL'
        ##    1  'Open window'
        ##    2  'Drink'
        ##    3  'Water plant'
        ##    4  'Close window'
        ##    5  'Cut'
        ##    6  'Chop'
        ##    7  'Stir'
        ##    8  'Book'
        ##    9  'Forehand'
        ##    10 'Backhand'
        ##    11 'Smash'
        
    classes = np.array([1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12])
    
    ordinal_labels = np.zeros((all_labels_shots.shape[0]))
    
    for i in range(all_labels_shots.shape[0]):
        k = np.where(classes == all_labels_shots[i])
        ordinal_labels[i]= k[0]

    ## split into 75% for train and 25% for test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_data_shots,
                                                        ordinal_labels,
                                                        test_size=0.2,
                                                        random_state=7)
    
    """
    above code maps each class label to a number between 0 and 10
    (This will be useful when you want to apply one-hot encoding using np_utils (from (keras.utils)) )
    """
    #### Saving Datasets ###
    np.save("train_data.npy", X_train)
    np.save("train_labels.npy", y_train)
    print("Info: Shape of train data = ",X_train.shape)
    np.save("test_data.npy", X_test)
    np.save("test_labels.npy", y_test)
    print("Info: Shape of test data = ",X_test.shape)
    return

##########
# Select Subject
create_sections(subject=1)