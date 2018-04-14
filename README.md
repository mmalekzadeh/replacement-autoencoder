# Replacement AutoEncoder (RAE)
## Keras Implementation of the Paper ["Replacement AutoEncoder: A Privacy-Preserving Algorithm for Sensory Data Analysis".](https://arxiv.org/abs/1710.06564)

## An example of the whole process on Skoda dataset.
Codes and files are available under "skoda" folder.

### Description of Skoda Dataset

This dataset describes the activities of assembly-line workers in a car production environment. They consider the recognition of 11 activity classes performed in one of the quality assurance checkpoints of the
production plan. In their study, one subject wore 19 3D accelerometers on both arms and perform a set of experiments using sensors placed on the two arms of a tester (10 sensors on the right arm and 9 sensor on the left arm). The original sample rate of this dataset was 98 Hz, but it was decimated to 30 Hz for comparison purposes with the other two datasets. The Skoda dataset has been employed to evaluate deep learning techniques in sensor networks, which makes it a proper dataset to evaluate our proposed Replacement AutoEncoder framework.

More details:  http://www.ife.ee.ethz.ch/research/activity-recognition-datasets.html

### Module 1: sm1_creating_sections.py
Skoda contain two time-series, one collected from **left** hand of data subject and another from **right** hand. Here, we build training and test datasets from time-series. 

Look at **Experimental Settings** section of the paper for more details.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September 2017

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
    
    ## split into 80% for train and 20% for test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_data_shots,
                                                        ordinal_labels,
                                                        test_size=0.2,
                                                        random_state=8)
    
    # ## normalize each sensor’s data to have a zero mean and unity standard deviation.
    for sid in range(X_train.shape[1]):
        mean = (np.mean(X_train[:,sid,:]))
        std  = (np.std(X_train[:,sid,:]))
        X_train[:,sid,:] -= mean
        X_test[:,sid,:] -= mean
        X_train[:,sid,:] /= std
        X_test[:,sid,:] /= std

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
create_sections(hand = "left")

```

    Info: Shape of train data =  (61945, 54, 30)
    Info: Shape of test data =  (15487, 54, 30)


### Module 2: sm2_server_cnn.py
This module trains a ConvNet model for using as the Server model.
Look at **Experimental Settings** section of the paper for more details.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in  September  2017
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
num_of_epochs = 10 ## CHOOSE NUMBER OF EPOCHES HERE
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
```

    Using TensorFlow backend.


    Epoch 1/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.7236 - acc: 0.7576
    Epoch 2/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.3893 - acc: 0.8709
    Epoch 3/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.3195 - acc: 0.8948
    Epoch 4/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.2791 - acc: 0.9084
    Epoch 5/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.2583 - acc: 0.9161
    Epoch 6/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.2431 - acc: 0.9201
    Epoch 7/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.2309 - acc: 0.9252
    Epoch 8/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.2220 - acc: 0.9267
    Epoch 9/10
    61945/61945 [==============================] - 95s 2ms/step - loss: 0.2120 - acc: 0.9301
    Epoch 10/10
    61945/61945 [==============================] - 96s 2ms/step - loss: 0.2020 - acc: 0.9343
    15487/15487 [==============================] - 8s 487us/step
    
     ~~~ Result: 
    acc: 95.08%
    Saved model to disk


### Module 3: sm3_build_wbg_datasets.py
Here we create three subsets from dataset: White, Gray, and Black.
These datasets are used for training RAE.
Look at **Experimental Settings** section of the paper for more details.


```python
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
white_list=np.array([4,8,9,10])
black_list=np.array([1,5,6,7])
gray_list=np.array([0,2,3])
#####################################

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
```

    Info: inference #0 is gray-listed
    Info: inference #1 is black-listed
    Info: inference #2 is gray-listed
    Info: inference #3 is gray-listed
    Info: inference #4 is white-listed
    Info: inference #5 is black-listed
    Info: inference #6 is black-listed
    Info: inference #7 is black-listed
    Info: inference #8 is white-listed
    Info: inference #9 is white-listed
    Info: inference #10 is white-listed
    Info: Size of white-listed training data is 19397
    Info: Size of black-listed training data is 15775
    Info: Size of gray-listed training data is 26773
    Info: Size of white-listed testing data is 4779
    Info: Size of black-listed testing data is 3923
    Info: Size of wgray-listed testing data is 6785
    Info: all lists are saved
    Info: lists of inferences are saved


### Module 4: sm4_replacement_autoencoder.py
In this modul we define RAE and train it using subsets created.


```python
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
num_of_epochs = 10

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
```

    Train on 61945 samples, validate on 15487 samples
    Epoch 1/10
    61945/61945 [==============================] - 63s 1ms/step - loss: 0.3700 - val_loss: 0.3073
    Epoch 2/10
    61945/61945 [==============================] - 63s 1ms/step - loss: 0.3055 - val_loss: 0.2977
    Epoch 3/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2958 - val_loss: 0.2918
    Epoch 4/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2905 - val_loss: 0.2935
    Epoch 5/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2868 - val_loss: 0.2956
    Epoch 6/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2838 - val_loss: 0.2844
    Epoch 7/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2828 - val_loss: 0.2830
    Epoch 8/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2807 - val_loss: 0.2811
    Epoch 9/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2798 - val_loss: 0.2852
    Epoch 10/10
    61945/61945 [==============================] - 62s 1ms/step - loss: 0.2787 - val_loss: 0.2820
    Saved model to disk
    Info: Transformed data are saved


### Modul 5: sm5_evaluation.py
Finally, we compare the output of RAE (transformed data) with the original data to see utility-privacy tradeoff.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in September  2017
@author: mmalekzadeh
"""
import numpy as np
from keras.models import model_from_json
from keras import backend as K

def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos

     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos

     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
     return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

### Global Variables ###
## Load Lists of Inferences
all_inferences = np.load("all_inferences.npy")
white_list = np.asarray(all_inferences[0].tolist()[0])
black_list = np.asarray(all_inferences[1].tolist()[0])
gray_list = np.asarray(all_inferences[2].tolist()[0])
num_classes = len(white_list)+len(black_list)+len(gray_list)
## Load Original Test Data of each list
## w -> white, b -> black, g -> gray
o_w_test_data = np.load("data_test_white.npy")
o_b_test_data = np.load("data_test_black.npy")
o_g_test_data = np.load("data_test_gray.npy")
o_w_test_data = np.reshape(o_w_test_data,
                           (len(o_w_test_data),
                            o_w_test_data.shape[1],
                            o_w_test_data.shape[2], 1))
o_b_test_data = np.reshape(o_b_test_data,
                           (len(o_b_test_data),
                            o_b_test_data.shape[1],
                            o_b_test_data.shape[2], 1))
o_g_test_data = np.reshape(o_g_test_data,
                           (len(o_g_test_data),
                            o_g_test_data.shape[1],
                            o_g_test_data.shape[2], 1))
## Load Transformed Test Data of each list
tr_w_test_data = np.load("transformed_w_test_data.npy")
tr_b_test_data = np.load("transformed_b_test_data.npy")
tr_g_test_data = np.load("transformed_g_test_data.npy")

## Load Labels of each list
w_test_label = np.load("label_test_white.npy")
b_test_label = np.load("label_test_black.npy")
g_test_label = np.load("label_test_gray.npy")


## Build one-hot codes for each list
y_white = np.zeros((w_test_label.shape[0], num_classes))
for i in range(w_test_label.shape[0]):
    y_white[i, int(w_test_label[i])] = 1

y_black = np.zeros((b_test_label.shape[0], num_classes))
for i in range(b_test_label.shape[0]):
    y_black[i, int(b_test_label[i])] = 1

y_gray = np.zeros((g_test_label.shape[0], num_classes))
for i in range(g_test_label.shape[0]):
    y_gray[i, int(g_test_label[i])] = 1

# load json and create model
json_file = open('server_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("server_cnn_model_weights.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy',precision ,recall])

## Evaluate Original Data          
o_w_scores = loaded_model.evaluate(o_w_test_data, y_white, verbose=1)  
o_b_scores = loaded_model.evaluate(o_b_test_data, y_black, verbose=1) 
o_g_scores = loaded_model.evaluate(o_g_test_data, y_gray, verbose=1)  
## Predict Original Data
o_w_predict = loaded_model.predict(o_w_test_data, verbose=1)
o_b_predict = loaded_model.predict(o_b_test_data, verbose=1)
o_g_predict = loaded_model.predict(o_g_test_data, verbose=1)

## Evaluate Transformed Data
tr_w_scores = loaded_model.evaluate(tr_w_test_data, y_white, verbose=1)  
tr_b_scores = loaded_model.evaluate(tr_b_test_data, y_black, verbose=1) 
tr_g_scores = loaded_model.evaluate(tr_g_test_data, y_gray, verbose=1)  
## Predict Transformed Data
tr_w_predict = loaded_model.predict(tr_w_test_data, verbose=1)
tr_b_predict = loaded_model.predict(tr_b_test_data, verbose=1)
tr_g_predict = loaded_model.predict(tr_g_test_data, verbose=1)

print("\n ~~~ Result: F1-Socre on Original Data:")
print("\n on white-listed: %.2f%%"%
      ((2*((o_w_scores[2]*o_w_scores[3])/(o_w_scores[2]+o_w_scores[3])))*100))
print("\n on black-listed %.2f%%"%
      ((2*((o_b_scores[2]*o_b_scores[3])/(o_b_scores[2]+o_b_scores[3])))*100))
print("\n on gray-listed %.2f%%"%
      ((2*((o_g_scores[2]*o_g_scores[3])/(o_g_scores[2]+o_g_scores[3])))*100))

print("\n ~~~ Result: F1-Socre on Transformed Data:")
print("\n on white-listed: %.2f%%"%
      ((2*((tr_w_scores[2]*tr_w_scores[3])/(tr_w_scores[2]+tr_w_scores[3])))*100))
print("\n on black-listed %.2f%%"%
      ((2*((tr_b_scores[2]*tr_b_scores[3])/(tr_b_scores[2]+tr_b_scores[3])))*100))
print("\n on gray-listed %.2f%%"%
      ((2*((tr_g_scores[2]*tr_g_scores[3])/(tr_g_scores[2]+tr_g_scores[3])))*100))


#
########### Calculating Confusion Matrix ###########
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=18,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

o_w_pred = np.argmax(o_w_predict,axis=1)
o_b_pred = np.argmax(o_b_predict,axis=1)
o_g_pred = np.argmax(o_g_predict,axis=1)
tr_w_pred = np.argmax(tr_w_predict,axis=1)
tr_b_pred = np.argmax(tr_b_predict,axis=1)
tr_g_pred = np.argmax(tr_g_predict,axis=1)

w_true = np.zeros(o_w_pred.shape[0])
b_true = np.zeros(o_b_pred.shape[0])
g_true = np.zeros(o_g_pred.shape[0])

for i in range(o_w_pred.shape[0]):
    w_true[i]= 1
    if o_w_pred[i] in gray_list:
        o_w_pred[i]= 2
    elif o_w_pred[i] in white_list:
        o_w_pred[i]= 1
    else:
        o_w_pred[i]= 0

for i in range(o_b_pred.shape[0]):
    b_true[i]= 0
    if o_b_pred[i] in gray_list:
        o_b_pred[i]= 2
    elif o_b_pred[i] in white_list:
        o_b_pred[i]= 1
    else:
        o_b_pred[i]= 0

for i in range(o_g_pred.shape[0]):
    g_true[i]= 2
    if o_g_pred[i] in gray_list:
        o_g_pred[i]= 2
    elif o_g_pred[i] in white_list:
        o_g_pred[i]= 1
    else:
        o_g_pred[i]= 0


for i in range(tr_w_pred.shape[0]):
    if tr_w_pred[i] in gray_list:
        tr_w_pred[i]= 2
    elif tr_w_pred[i] in white_list:
        tr_w_pred[i]= 1
    else:
        tr_w_pred[i]= 0

for i in range(tr_b_pred.shape[0]):
    if tr_b_pred[i] in gray_list:
        tr_b_pred[i]= 2
    elif tr_b_pred[i] in white_list:
        tr_b_pred[i]= 1
    else:
        tr_b_pred[i]= 0

for i in range(tr_g_pred.shape[0]):
    if tr_g_pred[i] in gray_list:
        tr_g_pred[i]= 2
    elif tr_g_pred[i] in white_list:
        tr_g_pred[i]= 1
    else:
        tr_g_pred[i]= 0


class_names =["B", "W", "G"]
ycf_test = np.append(w_true, g_true, axis=0)
ycf_test = np.append(ycf_test, b_true, axis=0)
ycf_o_pred = np.append(o_w_pred, o_g_pred, axis=0)
ycf_o_pred = np.append(ycf_o_pred, o_b_pred, axis=0)
ycf_tr_pred = np.append(tr_w_pred, tr_g_pred, axis=0)
ycf_tr_pred = np.append(ycf_tr_pred, tr_b_pred, axis=0)

## Compute confusion matrix for Original Data
o_cnf_matrix = confusion_matrix(ycf_test, ycf_o_pred)
np.set_printoptions(precision=3)
## Plot non-normalized confusion matrix
plot_confusion_matrix(o_cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion Matrix of Original Data')
plt.savefig('OCF.pdf',bbox_inches='tight')

plt.gcf().clear()

## Compute confusion matrix for Transformed Data
tr_cnf_matrix = confusion_matrix(ycf_test, ycf_tr_pred)
np.set_printoptions(precision=3)
## Plot non-normalized confusion matrix
plot_confusion_matrix(tr_cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion Matrix of Transformed Data')
plt.savefig('TrCF.pdf',bbox_inches='tight')
```

    Loaded model from disk
    4779/4779 [==============================] - 2s 487us/step
    3923/3923 [==============================] - 2s 481us/step
    6785/6785 [==============================] - 3s 502us/step
    4779/4779 [==============================] - 2s 465us/step
    3923/3923 [==============================] - 2s 543us/step
    6785/6785 [==============================] - 3s 470us/step
    4779/4779 [==============================] - 2s 468us/step
    3923/3923 [==============================] - 2s 546us/step
    6785/6785 [==============================] - 3s 460us/step
    4779/4779 [==============================] - 2s 501us/step
    3923/3923 [==============================] - 2s 488us/step
    6785/6785 [==============================] - 3s 461us/step
    
     ~~~ Result: F1-Socre on Original Data:
    
     on white-listed: 96.98%
    
     on black-listed 93.60%
    
     on gray-listed 94.63%
    
     ~~~ Result: F1-Socre on Transformed Data:
    
     on white-listed: 94.25%
    
     on black-listed 0.51%
    
     on gray-listed 92.28%
    Normalized confusion matrix
    [[0.965 0.001 0.034]
     [0.002 0.97  0.028]
     [0.018 0.012 0.97 ]]
    Normalized confusion matrix
    [[1.300e-02 2.549e-04 9.867e-01]
     [4.813e-03 9.464e-01 4.875e-02]
     [6.780e-03 1.238e-02 9.808e-01]]

