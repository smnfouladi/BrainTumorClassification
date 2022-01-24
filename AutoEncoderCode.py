import tensorflow.compat.v1 as tf
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard,ModelCheckpoint

import glob
import os

import cv2
from math import sin, cos, radians
import random
import numpy as np
from keras.models import Sequential,load_model,model_from_json


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Cropping2D,Activation,ZeroPadding2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import matplotlib.pyplot as plt
from statistics import mean
import numpy as np, pandas as pd, io, csv

##compute_class_weight Modules
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#confusion_matrix Modules

from sklearn.metrics import confusion_matrix


## Precision/Recall/F1_score AND Roc curve Modules
from scipy import interp
import itertools    
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


#Classifier Lib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


##########################################################################################################################

    
Datapath='/content/AllData'

####Load target
target=[]

for img_filename in os.listdir(Datapath):
    imgtype=img_filename.split("_",1)[0]
    if (imgtype=='glioma'):
        target.append('0')
    if (imgtype=='meningioma'):
        target.append('1')
    if (imgtype=='pituitary'):
        target.append('2')
    if (imgtype=='no'):
        target.append('3')


#print(target)        
####Load Data        

arrayofdata_=[]
arrayofdata=[]

for filename in glob.glob('/content/AllData/*.jpg'):
    img=cv2.imread(filename)
    img=cv2.resize(img, (80, 80), interpolation=cv2.INTER_LINEAR) 
    img=np.array(img)    
    inputdata = np.reshape(img, (img.shape[0],img.shape[1],img.shape[2]))
    inputdata=np.array(inputdata)
    arrayofdata.append(inputdata.tolist())
    arrayofdata_=arrayofdata

arrayofdata_ = np.array(arrayofdata_)
one_hot_labels = to_categorical(target, num_classes=4)

#print(one_hot_labels)
    
x_train, x_test, y_train, y_test = train_test_split(arrayofdata_,
                                                          one_hot_labels,
                                                          test_size=0.2,shuffle=True,
                                                          random_state=42)

print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],x_train.shape[2],3))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],x_test.shape[2],3))



#Reshape Data for Confusion_matrix
x_train1 = np.reshape(x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[2]*3))
x_test1 = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]*3))
y_train1 = np.argmax(y_train, axis=1)
y_test1 = np.argmax(y_test, axis=1)


def AutoEncoder_Network():

    input_img = Input(shape=(x_train.shape[1],x_train.shape[2],3)) 

    #مدل آموزش
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    print(encoded)
     
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.1)(x)
      
    decoded= Conv2D(128, (2, 2), activation='relu', padding='same')(x)

    autoencoder = Model(input_img, decoded)   
        
    #کلاس بندی با استفاده از خروجی اینکدر
    

    y = Conv2D(64, (2, 2), padding = 'same', activation='relu' , name = 'conv8')(encoded)    
    y = Conv2D(64, (2, 2),  padding = 'same',activation='relu' )(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D((2, 2), name = 'max7')(y)
    y = Dropout(0.1)(y)



    y = Flatten()(y)

    
    y1 = Dense(4, activation='softmax')(y)     
    classifier= Model(inputs=autoencoder.input, outputs=y1)     
 
    classifier.summary()

    classifier.compile(Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
      


    y_integers = np.argmax(one_hot_labels, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))

    history = classifier.fit(x_train, y_train, 
                      verbose=2, 
                      class_weight = d_class_weights,
                      batch_size = 16,
                      validation_data=(x_test,y_test),
                      epochs = 100)


    print(mean(history.history['accuracy']))
    print(mean(history.history['val_accuracy']))

    print('losssssssssss')
    print(mean(history.history['loss']))
    print(mean(history.history['val_loss']))


    #Model Loss
    plt.plot(history.history['loss'], linewidth=2, label='Train')
    plt.plot(history.history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')    
    plt.show()

    #Model ACC


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    
    return classifier
#################################################Confusion_matrix Code#########################################


Net=AutoEncoder_Network()
Net = Net.predict(x_test)
classification=np.argmax(Net)
y_pred_2=Net.argmax(axis=-1)

class_names=["Glioma","Meningioma","Pituitary","Healthy"]




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  

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
                color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

cnf_matrix = confusion_matrix(y_test1, y_pred_2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()




#################################################Precision/Recall/F1_score AND Roc curve#########################################
  



# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(4):
    precision[i], recall[i], _ = precision_recall_curve(y_test[i],Net[i])
                                                            
    average_precision[i] = average_precision_score(y_test[i], Net[i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        Net.ravel())
average_precision["micro"] = average_precision_score(y_test, Net,
                                                          average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    

precision_=mean(precision["micro"])
recall_=mean(recall["micro"])
f_Score=2*((precision_*recall_)/(precision_+recall_+K.epsilon()))
print(classification_report(y_test, Net.round(), target_names=class_names))
    
n_classes =4
    

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Net[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Net.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
              color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
              label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
