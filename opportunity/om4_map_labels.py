#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  September  2017

    This module map each class label to a number between 0 and 17
    (This will be useful when you want to apply one-hot encoding...
    using np_utils (from (keras.utils)) )

@author: mmalekzadeh
"""
import numpy as np

train_label = np.load("sections_label_train.npy")
test_label = np.load("sections_label_test.npy")

## There are 18 Gestures classes in dataset
## Respectively :  0:Null, 1:Open_Door1, 2:Open_Door2,  3:Close_Door1, 4:Close_Door2,
##                 5:Open_Fridge,  6:Close_Fridge, 7:Open_Dishwasher, 8:Close_Dishwasher, 
##                 9:Open_Drawer1, 10:Close_Drawer1, 11:Open_Drawer2,12:Close_Drawer2, 
##                 13:Open_Drawer3, 14:Close_Drawer3,
##                 15:Clean_Table, 16:Drink_Cup, 17:Toggle_Switch 
classes = np.array([0, 506616, 506617, 504616, 504617,
                    506620, 504620, 506605, 504605,
                    506619, 504619, 506611, 504611, 
                    506608, 504608,
                    508612, 507621, 505606])
ordinal_train_label = np.zeros((train_label.shape[0]))
ordinal_test_label = np.zeros((test_label.shape[0]))

for i in range(train_label.shape[0]):
    k = np.where(classes == train_label[i])
    ordinal_train_label[i]= k[0]

for i in range(test_label.shape[0]):
    k= np.where(classes == test_label[i])
    ordinal_test_label[i] = k[0]
    
np.save("sections_label_train.npy", ordinal_train_label)
print("Ordinal Training Labels are saved")
np.save("sections_label_test.npy", ordinal_test_label)
print("Ordinal Testing Labels are saved")
