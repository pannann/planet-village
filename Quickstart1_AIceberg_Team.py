#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from skimage.feature import hog 
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[ ]:





# In[ ]:





# In[2]:


classes_train = sorted(list(filter(lambda x: os.path.isdir(r'C:\Users\Ahmed Phd Concordia\Documents\Climate change AI/train/' + x), os.listdir(r'C:\Users\Ahmed Phd Concordia\Documents\Climate change AI\train'))))
#classes_val = sorted(list(filter(lambda x: os.path.isdir('./val/' + x), os.listdir('./val'))))
#assert classes_train == classes_val


# In[3]:


classes_train


# In[4]:


sample_impaths = list(filter(lambda x: x.endswith('JPG') or x.endswith('jpg'), os.listdir(r'C:\Users\Ahmed Phd Concordia\Documents\Climate change AI/train/' + classes_train[0])))


# In[5]:


x = plt.imread(os.path.join(r'C:\Users\Ahmed Phd Concordia\Documents\Climate change AI/train', classes_train[0], sample_impaths[0]))


# In[6]:


plt.imshow(x)
print(type(x))


# In[7]:


default_image_size = tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)  
            image = img_to_array(image)
            return image
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[8]:


def image_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# In[9]:


def convert_image_to_clor(image_dir):
    try:
        image = cv2.cvt(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[10]:



image_list, label_list, hu_moments, hog_list= [], [], [], []
directory_root = r'C:\Users\Ahmed Phd Concordia\Documents\Climate change AI/train/'



root_dir = listdir(directory_root)
root_dir.sort()
for directory in root_dir :
    # remove .DS_Store from list
    if directory == ".DS_Store" :
        root_dir.remove(directory)

#for plant_folder in root_dir :
 #   plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")



for plant_disease_folder in root_dir:
    plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}")

    for single_plant_disease_image in plant_disease_image_list :
        if single_plant_disease_image == ".DS_Store" :
            plant_disease_image_list.remove(single_plant_disease_image)

    for image in plant_disease_image_list[:200]:
        image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
        if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
            array_image = convert_image_to_array(image_directory)
            image_list.append(array_image)
            hu_moments.append(image_hu_moments(array_image))
            hog_list.append(hog(array_image))
            label_list.append(plant_disease_folder)
            
  


# In[11]:


image_size = len(image_list)
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 


# In[13]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
model = Sequential()
inputShape = (256, 256, 3)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))


# In[14]:


model.summary()


# In[15]:


EPOCHS = 25
INIT_LR = 1e-3

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")


# In[16]:


BS=32
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )


# In[18]:


model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

y_predict_label = []
for image in y_predict:
    m = np.amax(image)
    l = list(image)
    ind = l.index(m)
    y_predict_label.append(classes_train[ind])

y_test_labels = []
for y in y_test:
    l = list(y)
    i = l.index(1)
    y_test_labels.append(classes_train[i])

    
labels = set(y_test_labels)

def display_confusion_matrix(y_test, dtree_predictions, labels):
    cm = confusion_matrix(y_test,dtree_predictions) 

  # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm, index = labels,columns = labels)
    plt.figure(figsize=(20,20))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
display_confusion_matrix(y_test_labels,y_predict_label, labels)


# In[33]:


image_size = len(image_list)
df=pd.DataFrame(list(zip(image_list,label_list)),columns=['image','label'])


# In[35]:


def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score


# In[37]:


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Fbeta')
    pyplot.plot(history.history['fbeta'], color='blue', label='train')
    pyplot.plot(history.history['val_fbeta'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

