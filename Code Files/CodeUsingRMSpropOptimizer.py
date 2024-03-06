# Importing the required Libraries!

# Importing Libraries Code Starts...
import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img

import cv2
import itertools
import pathlib
import warnings
from PIL import Image
from random import randint
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D

from distutils.dir_util import copy_tree, remove_tree
# Importing Libraries Code Ends...


# ------------------------------ TO REMOVE THE CHECKPOINT DIRECTORY MADE BY JUPYTER LAB----------------------------------

# Specify the path to the directory containing your data
data_directory = '../Alzheimer_Dataset/WHOLE_DATASET'

# Get a list of all subdirectories and files in the data directory
subdirectories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]

# Remove .ipynb_checkpoints directories if present
for subdirectory in subdirectories:
    if subdirectory == '.ipynb_checkpoints':
        subdirectory_path = os.path.join(data_directory, subdirectory)
        try:
            os.rmdir(subdirectory_path)
            print(f"Removed '{subdirectory}' directory.")
        except OSError as e:
            print(f"Error while removing '{subdirectory}' directory: {e}")
            
# Print the list of directories in the main dataset directory....
print(os.listdir(data_directory))

# Print the version of tensorflow....
print("TensorFlow Version:", tf.__version__)


# Splitting the dataset to into Validation, Train and Test in a directory 'Output' to preprocess the images!

# Splitting of directories code starts........
#!pip install split-folders
import splitfolders
splitfolders.ratio(data_directory, output="Dataset_After_Splitting", seed=1345, ratio=(.8, 0.1,0.1))
# Splitting of directories code ends........


# Preprocessing the dataset!

# Preprocessing dataset code starts...........
IMG_HEIGHT = 128
IMG_WIDTH = 128

training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./output/train",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
)

testing_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./output/test",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./output/val",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
)
#  Preprocessing dataset code ends...........


# Print the class names of the training directory....
class_names = training_dataset.class_names
print(class_names)
training_dataset


# To inspect a small sample of the training data to gain a visual understanding of how the images in the dataset are labeled.

# Code for visualization of sample images starts...........
plt.figure(figsize=(7, 7))
for images, labels in training_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# Code for visualization of sample images ends...........


# Developing the Model!

# Code for development of the model starts...........
model = keras.models.Sequential()
model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)))
model.add(keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Dropout(0.20))

model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu",kernel_initializer="he_normal"))
model.add(keras.layers.Dense(64,"relu"))
model.add(keras.layers.Dense(6,"softmax"))
print("--Above Code Executed & Model Developed Successfully--")
# Code for development of the model ends...........


# Compiling the Model!

# Model Compilation code starts......
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "RMSprop",metrics=["accuracy"])
print("---Model Compiled Successfully---")
# Model Compilation code ends......


# Code for Model Summary!
model.summary()

# Code for training the model!
ModelTraining = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=20
)

# Code to retrieve and display key training and validation metrics
# such as accuracy and loss from a previously trained neural network model!

# Code to retrieve and display key training and validation metrics starts.......
get_ac = ModelTraining.history['accuracy']
print("THE ACCURACY IS:",round(get_ac[-1]*100,2))
print("_____________________")
get_los = ModelTraining.history['loss']
print("THE LOSS IS:",round(get_los[-1]*100,2))
print("_____________________")
val_acc = ModelTraining.history['val_accuracy']
print("THE Validation Accuracy IS:",round(val_acc[-1]*100,2))
print("_____________________")
val_loss = ModelTraining.history['val_loss']
print("THE Validation Loss IS:",round(val_loss[-1]*100,2))
# Code to retrieve and display key training and validation metrics ends.......
