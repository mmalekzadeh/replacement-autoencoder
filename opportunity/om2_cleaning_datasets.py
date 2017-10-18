#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 2017

This module clean training and testing datasets.
They contain some missing data in some columns.
1) Filling: We remove any column that contains more than TotalMiss% of missing
   values. In others columns, we subsituite any missing value
   using Linear Interpolation on the corresponding column.  

2) Normalization: We normalize each sensor’s data to have
   a zero mean and unity standard deviation.

@author: mmalekzadeh
"""
import numpy as np
from scipy import stats
#####################################
# fix random seed for reproducibility
np.random.seed(2017)
#####################################
 
### Global Variables ###
## To import training and testing dataset produced by m1_creating_datasets.py
train_data = np.load("training_dataset.npy")
test_data = np.load("testing_dataset.npy")

### To Fill in Missing Values ###
def fill_missing_data(total_miss):
    global train_data
    global test_data
    size_train = train_data.shape[0]
    size_test = test_data.shape[0]

    ## number of features
    ## The last two columns are labels of data, they don't need cleaning
    num_of_cols = train_data.shape[1]-2
    
    ## removing rows which have more than "total_miss"% missing values
    reduced_train_data = np.zeros(train_data.shape)
    row_acceptance_rate = num_of_cols*total_miss    

    jtr = 0
    for i in range(size_train):
        num_of_nulls = np.count_nonzero(np.isnan(train_data[i, :]))
        if  num_of_nulls < row_acceptance_rate:
            reduced_train_data[jtr] = train_data[i]
            jtr = jtr+1    
    print("Info: number of rows removed in train data: ", size_train-jtr)
    
    reduced_test_data = np.zeros(test_data.shape)
    jte = 0
    for i in range(size_test):
        num_of_nulls = np.count_nonzero(np.isnan(test_data[i, :]))
        if num_of_nulls < row_acceptance_rate:
            reduced_test_data[jte] = test_data[i]
            jte = jte+1
    print("Info: number of rows removed in test data: ", size_test-jte)
    
    reduced_train_data = reduced_train_data[0:jtr, :]
    reduced_test_data = reduced_test_data[0:jte, :]

    ## Concatenating Datasets
    ## We want to fill missing values in all datasets at the same time.
    all_data = np.concatenate((reduced_train_data, reduced_test_data))
    size_all = all_data.shape[0]
    col_acceptance_rate = size_all*total_miss    

    ## removing columns which have more than "total_miss"% missing values
    count = 0
    i = 0
    while count < num_of_cols:
        num_of_nulls = np.count_nonzero(np.isnan(all_data[:,i]))
        if  num_of_nulls > col_acceptance_rate:
            all_data = np.delete(all_data, i, 1)
            print("Info Column {} is removed".format(count))        
        else:
            i = i+1
        count = count+1
    print("Info: remaining {}% NaNs in all data"
          .format((np.count_nonzero(np.isnan(all_data))*100)//np.prod(all_data.shape)))

    ## Filling in missing values using Linear Interpolation
    ## number of remaining features
    num_of_cols = all_data.shape[1]-2
    for i in range(1,num_of_cols):
        temp = all_data[:,i]
        nans, xf = np.isnan(temp), lambda z: z.nonzero()[0]
        temp[nans] = np.interp(xf(nans), xf(~nans), temp[~nans])
        all_data[:,i] = temp
    print("Info: remaining {} NaNs in all data (After Interpolation)"
          .format(np.count_nonzero(np.isnan(all_data))))

    ## Seperating Datasets
    train_data = all_data[0:jtr,:]
    test_data = all_data[jtr:jtr+jte,:]
    
    return;

### Normalize all the Columns ###
def normalize_all_columns():
    global train_data
    global test_data
    size_train = train_data.shape[0]
    size_test = test_data.shape[0]

    ## Concatenating Datasets
    ## We want both datasets to be normalized with the same parameters.
    all_data = np.concatenate((train_data, test_data))
    num_of_cols = all_data.shape[1]-2
    ## col 0 is time, the last two cols are classes ...
    ## so, from 1 to num_of_cols
    ## normalize each sensor’s data to have a zero mean and... 
    ## unity standard deviation.
    tmp_all_data = stats.zscore(all_data[:, 1:num_of_cols], axis=0, ddof=0)
    tmp_gesture_classes = all_data[:, -1]
    tmp_gesture_classes = np.expand_dims(tmp_gesture_classes,axis=1)
    all_data = np.concatenate((tmp_all_data,tmp_gesture_classes), axis=1)

    ## Seperating Datasets
    train_data = all_data[0:size_train, :]
    test_data = all_data[size_train:size_train+size_test, :]
    
    ### Saving Datasets ###
    np.save("clean_training_dataset.npy", train_data)
    print("Info: Cleaned and Normalized Training Dataset with shape {} is saved"
          .format(train_data.shape))
    np.save("clean_testing_dataset.npy", test_data)
    print("Info: Cleaned and Normalized Testing Dataset with shape {} is saved"
          .format(test_data.shape))
    return;

### Calling desired Moduls 
"""
    1) "fill_missing_data" remove each row/col which have more than total_miss of missing data.
    2) "normalize_all_columns()" normalize each sensor’s data to have a zero mean and unity standard deviation 
"""
total_miss = 0.3
fill_missing_data(total_miss)
normalize_all_columns()