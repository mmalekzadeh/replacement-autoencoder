#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created  September  2017
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
