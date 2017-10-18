#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September 2017

This module merges seperated files in the OPPORTUNITY Activity Recognition
DataSet.
It enables to create customized training and testing dataset. 

@author: mmalekzadeh
"""
import numpy as np

### Loading, Merging and Creating Training and Testing Datasets
def merge_opp_data(number_of_features, **options):
    train_data = np.zeros((0,number_of_features))
    test_data = np.zeros((0,number_of_features))    
    args_train = options.get("train")
    args_test = options.get("test")
    ## Subject 1
    #### Drill
    if 10 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S1-Drill.txt")))
        print("Info: S1 Drill as train")
        
    if 10 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S1-Drill.txt")))
        print("Info: S1 Drill as test")
        
    #### ADL1    
    if 11 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S1-ADL1.txt")))
        print("Info: S1 ADL1 as train")
    
    if 11 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S1-ADL1.txt")))
        print("Info: S1 ADL1 as test")
    
    #### ADL2    
    if 12 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S1-ADL2.txt")))
        print("Info: S1 ADL2 as train")
    
    if 12 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S1-ADL2.txt")))
        print("Info: S1 ADL2 as test")
        
    #### ADL3    
    if 13 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S1-ADL3.txt")))
        print("Info: S1 ADL3 as train")
    
    if 13 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S1-ADL3.txt")))
        print("Info: S1 ADL3 as test")    

    #### ADL4    
    if 14 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S1-ADL4.txt")))
        print("Info: S1 ADL4 as train")
    
    if 14 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S1-ADL4.txt")))
        print("Info: S1 ADL4 as test") 
        
    #### ADL5    
    if 15 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S1-ADL5.txt")))
        print("Info: S1 ADL5 as train")
    
    if 15 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S1-ADL5.txt")))
        print("Info: S1 ADL5 as test")     

    ## Subject 2
    #### Drill
    if 20 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S2-Drill.txt")))
        print("Info: S2 Drill as train")

    if 20 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S2-Drill.txt")))
        print("Info: S2 Drill as test")
        
    #### ADL1    
    if 21 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S2-ADL1.txt")))
        print("Info: S2 ADL1 as train")
    
    if 21 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S2-ADL1.txt")))
        print("Info: S2 ADL1 as test")
    
    #### ADL2    
    if 22 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S2-ADL2.txt")))
        print("Info: S2 ADL2 as train")
    
    if 22 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S2-ADL2.txt")))
        print("Info: S2 ADL2 as test")
        
    #### ADL3    
    if 23 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S2-ADL3.txt")))
        print("Info: S2 ADL3 as train")
    
    if 23 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S2-ADL3.txt")))
        print("Info: S2 ADL3 as test")    

    #### ADL4    
    if 24 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S2-ADL4.txt")))
        print("Info: S2 ADL4 as train")
    
    if 24 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S2-ADL4.txt")))
        print("Info: S2 ADL4 as test") 
        
    #### ADL5    
    if 25 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S2-ADL5.txt")))
        print("Info: S2 ADL5 as train")
    
    if 25 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S2-ADL5.txt")))
        print("Info: S2 ADL5 as test")
    
    ## Subject 3
    #### Drill
    if 30 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S3-Drill.txt")))
        print("Info: S3 Drill as train")

    if 30 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S3-Drill.txt")))
        print("Info: S3 Drill as test")
        
    #### ADL1    
    if 31 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S3-ADL1.txt")))
        print("Info: S3 ADL1 as train")
    
    if 31 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S3-ADL1.txt")))
        print("Info: S3 ADL1 as test")
    
    #### ADL2    
    if 32 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S3-ADL2.txt")))
        print("Info: S3 ADL2 as train")
    
    if 32 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S3-ADL2.txt")))
        print("Info: S3 ADL2 as test")
        
    #### ADL3    
    if 33 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S3-ADL3.txt")))
        print("Info: S3 ADL3 as train")
    
    if 33 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S3-ADL3.txt")))
        print("Info: S3 ADL3 as test")    

    #### ADL4    
    if 34 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S3-ADL4.txt")))
        print("Info: S3 ADL4 as train")
    
    if 34 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S3-ADL4.txt")))
        print("Info: S3 ADL4 as test") 
        
    #### ADL5    
    if 35 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S3-ADL5.txt")))
        print("Info: S3 ADL5 as train")
    
    if 35 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S3-ADL5.txt")))
        print("Info: S3 ADL5 as test")
    
    ## Subject 4
    #### Drill
    if 40 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S4-Drill.txt")))
        print("Info: S4 Drill as train")

    if 40 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S4-Drill.txt")))
        print("Info: S4 Drill as test")
        
    #### ADL1    
    if 41 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S4-ADL1.txt")))
        print("Info: S4 ADL1 as train")
    
    if 41 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S4-ADL1.txt")))
        print("Info: S4 ADL1 as test")
    
    #### ADL2    
    if 42 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S4-ADL2.txt")))
        print("Info: S4 ADL2 as train")
    
    if 42 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S4-ADL2.txt")))
        print("Info: S4 ADL2 as test")
        
    #### ADL3    
    if 43 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S4-ADL3.txt")))
        print("Info: S4 ADL3 as train")
    
    if 43 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S4-ADL3.txt")))
        print("Info: S4 ADL3 as test")    

    #### ADL4    
    if 44 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S4-ADL4.txt")))
        print("Info: S4 ADL4 as train")
    
    if 44 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S4-ADL4.txt")))
        print("Info: S4 ADL4 as test") 
        
    #### ADL5    
    if 45 in args_train:
        train_data = np.concatenate((train_data,np.loadtxt("S4-ADL5.txt")))
        print("Info: S4 ADL5 as train")
    
    if 45 in args_test:
        test_data = np.concatenate((test_data,np.loadtxt("S4-ADL5.txt")))
        print("Info: S4 ADL5 as test")
    
    ### Saving Datasets ###
    np.save("training_dataset.npy", train_data)
    print("Info: Training Dataset with shape {} is saved".format(train_data.shape))
    np.save("testing_dataset.npy", test_data)
    print("Info: Testing Dataset with shape {} is saved".format(test_data.shape))
    return;

### Choosing Desired Files for Making Dataset
"""
    Here you can choose your desired files
    to creat your training and testing dataset
    --> Example:
        merge_opp_data(number_of_features = 116,
                       train =[10,11,12,13,14],
                       test=[15])
        0 is for Drills and 1,2,3,4,5 are respectively for ADLs
        e.g. 23 means S2-ADL3 (data of activity #3 from subject #2)
    Each entry of OPPORTUNITY Dataset contains 116 columns
    [time, ...113 sensory values... , Locomotion, Gestures]
"""
merge_opp_data(number_of_features = 116,
               train = [10,11,12,13,14],
               test = [15]
               )