    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 2017

This module merges seperated files in OPPORTUNITY Activity Recognition Data Set.
It enables to create customized training and testing dataset. 

@author: mmalekzadeh
"""
import numpy as np
from scipy import stats
    
def create_sections(hand):
    if(hand == "left"):
        all_d = np.loadtxt("left_hand.txt", delimiter=',')
    else:
        all_d = np.loadtxt("right_hand.txt", delimiter=',')
        
    ## The original sample rate of this dataset was 98 Hz,
    ## but it was decimated to 30 Hz 
    ## for comparison purposes with the OPPORTUNITY dataset \cite{ordonez2016deep}
    all_d = all_d[::3]
    
    ## Sensory Data (10 sensors , each 6 values)(ecah column shows a recored)
    all_data = all_d[:,1:71]
    ## label for each entry (each recored has a label)
    all_labels = all_d[:,0]


    ## removing sensor ids
    for i in range(0,60,6):
        all_data = np.delete(all_data, i, 1)
    
    ##### ATTENTION #####
    ### Just For Left Hand: Remove data of Sensor numer 30
    if(hand == "left"):
        seg1 = all_data[:,0:48]
        seg2 = all_data[:,54:60]
        all_data = np.append(seg1, seg2 ,axis=1) 
    
    
    # ## normalize each sensor’s data to have a zero mean and unity standard deviation.
    all_data = stats.zscore(all_data, axis=0, ddof=0)
    
    all_data = all_data.T
    all_labels = all_labels.T
    
    ##  This Variable Defines the Size of Sliding Window
    ##  ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor) 
    sliding_window_size = 30
    
    ##  Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
    ##  ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
    step_size_of_sliding_window = 3
        
    size_features = all_data.shape[0]
    size_all_data = all_data.shape[1]
        
    number_of_shots = round(((size_all_data - sliding_window_size)/step_size_of_sliding_window))
        
    ##  Create a 3D matrix for Storing Snapshots  
    all_data_shots= np.zeros((number_of_shots , size_features , sliding_window_size ))
    all_labels_shots = np.zeros(number_of_shots)
    
    for i in range(0 ,(size_all_data) - sliding_window_size  , step_size_of_sliding_window):
        all_data_shots[i // step_size_of_sliding_window ] = all_data[0:size_features, i:i+sliding_window_size]
        all_labels_shots[i // step_size_of_sliding_window] = np.argmax(np.bincount(all_labels[i:i+sliding_window_size].astype(int)))
        
    """
        1) above code steps on time-series data and extract temporal snapshots.
        We build this new kind of dataset because its more flexible for Convolutional Neural Networks
    """
    
    ## There are 11 Gestures classes in dataset
    ## Respectively :
    ##    0 null class
    ##    1 write on notepad
    ##    2 open hood
    ##    3 close hood
    ##    4 check gaps on the front door
    ##    5 open left front door
    ##    6 close left front door
    ##    7 close both left door
    ##    8 check trunk gaps
    ##    9 open and close trunk
    ##    10 check steering wheel
    
    classes = np.array([32, 48, 49, 50,
                        51, 52, 53, 54,
                        55, 56, 57])
    
    ordinal_labels = np.zeros((all_labels_shots.shape[0]))
    
    for i in range(all_labels_shots.shape[0]):
        k = np.where(classes == all_labels_shots[i])
        ordinal_labels[i]= k[0]
    
    ## split into 75% for train and 25% for test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_data_shots,
                                                        ordinal_labels,
                                                        test_size=0.2,
                                                        random_state=8)
    
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
# Select Hand: "left" or "right"
create_sections(hand = "right")
