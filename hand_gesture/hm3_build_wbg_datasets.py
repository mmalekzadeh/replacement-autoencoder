    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September  2017

@author: mmalekzadeh
"""
import numpy as np
from sklearn.utils import shuffle

### Global Variables ###
train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")
test_label = np.load("test_labels.npy")
train_label = np.load("train_labels.npy")


### Choosing Inferences Combinations ###
white_list=np.array([1,2,3,4,9,10,11])
black_list=np.array([5,6,7,8])
gray_list=np.array([0])
########################################

num_classes = np.unique(train_label).shape[0] 

num_train, height, width = train_data.shape
num_test = test_data.shape[0] 
     
white_train_data = np.zeros((0, height, width))
black_train_data = np.zeros((0, height, width))
gray_train_data = np.zeros((0, height, width))

white_test_data = np.zeros((0, height, width))
black_test_data = np.zeros((0, height, width))
gray_test_data = np.zeros((0, height, width))

white_train_label = np.zeros((0))
black_train_label = np.zeros((0))
gray_train_label = np.zeros((0))

white_test_label = np.zeros((0))
black_test_label = np.zeros((0))
gray_test_label = np.zeros((0))

for i in range(num_classes):
    if i in black_list :
        print("Info: inference #{} is black-listed".format(i))
        black_train_data = np.append(black_train_data , train_data[train_label == i], axis=0)
        black_train_label = np.append(black_train_label , train_label[train_label == i], axis=0)
        
        black_test_data = np.append(black_test_data , test_data[test_label == i], axis=0)
        black_test_label = np.append(black_test_label , test_label[test_label == i], axis=0)
    
    elif i in gray_list: 
        print("Info: inference #{} is gray-listed".format(i))
        gray_train_data = np.append(gray_train_data , train_data[train_label == i], axis=0)
        gray_train_label = np.append(gray_train_label , train_label[train_label == i], axis=0)
    
        gray_test_data = np.append(gray_test_data , test_data[test_label == i], axis=0)
        gray_test_label = np.append(gray_test_label , test_label[test_label == i], axis=0)
    
    elif i in white_list:
        print("Info: inference #{} is white-listed".format(i))
        white_train_data = np.append(white_train_data , train_data[train_label == i], axis=0)
        white_train_label = np.append(white_train_label , train_label[train_label == i], axis=0)
        
        white_test_data = np.append(white_test_data , test_data[test_label == i], axis=0)
        white_test_label = np.append(white_test_label , test_label[test_label == i], axis=0)

white_train_data, white_train_label = shuffle(white_train_data, white_train_label, random_state=0)
white_test_data, white_test_label = shuffle(white_test_data, white_test_label, random_state=0)

black_train_data, black_train_label = shuffle(black_train_data, black_train_label, random_state=0)
black_test_data, black_test_label = shuffle(black_test_data, black_test_label, random_state=0)

gray_train_data, gray_train_label = shuffle(gray_train_data, gray_train_label, random_state=0)
gray_test_data, gray_test_label = shuffle(gray_test_data, gray_test_label, random_state=0)
     
### Saving Datasets ###
np.save("data_train_white.npy", white_train_data)
np.save("label_train_white.npy", white_train_label)
print("Info: Size of white-listed training data is {}"
      .format(len(white_train_label)))

np.save("data_train_black.npy", black_train_data)
np.save("label_train_black.npy", black_train_label)
print("Info: Size of black-listed training data is {}"
      .format(len(black_train_label)))

np.save("data_train_gray.npy", gray_train_data)
np.save("label_train_gray.npy", gray_train_label)
print("Info: Size of gray-listed training data is {}"
      .format(len(gray_train_label)))

np.save("data_test_white.npy", white_test_data)
np.save("label_test_white.npy", white_test_label)
print("Info: Size of white-listed testing data is {}"
      .format(len(white_test_label)))

np.save("data_test_black.npy", black_test_data)
np.save("label_test_black.npy", black_test_label)
print("Info: Size of black-listed testing data is {}"
      .format(len(black_test_label)))

np.save("data_test_gray.npy", gray_test_data)
np.save("label_test_gray.npy", gray_test_label)
print("Info: Size of wgray-listed testing data is {}"
      .format(len(gray_test_label)))
print("Info: all lists are saved")

## Saving Lists
all_inferences = [[white_list], [black_list], [gray_list]]
np.save("all_inferences.npy", all_inferences)
print("Info: lists of inferences are saved")
