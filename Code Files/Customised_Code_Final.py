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
print(os.listdir("../Alzheimer_Dataset/WHOLE_DATASET"))

# Print the version of tensorflow....
print("TensorFlow Version:", tf.__version__)


# Splitting the dataset to into Validation, Train and Test in a directory 'Output' to preprocess the images!

# Splitting of directories code starts........
!pip install split-folders
import splitfolders
splitfolders.ratio('../Alzheimer_Dataset/WHOLE_DATASET', output="Dataset_After_Splitting", seed=1345, ratio=(.8, 0.1,0.1))
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
              optimizer = "Adam",metrics=["accuracy"])
print("--Model Compiled Successfully--")
# Model Compilation code ends......


# Code for Model Summary!
model.summary()


# Code for training the model!
ModelTraining = model.fit(training_dataset,validation_data=validation_dataset,epochs=20, batch_size=64, verbose=1)


# Code to retrieve and display key training and validation metrics
# such as accuracy and loss from a previously trained neural network model!

# Code to retrieve and display key training and validation metrics starts.......
get_ac = ModelTraining.history['accuracy']
print('Accuracy:',get_ac)
get_los = ModelTraining.history['loss']
print('Loss:',get_los)
val_acc = ModelTraining.history['val_accuracy']
print('Validation Accuracy:',val_acc)
val_loss = ModelTraining.history['val_loss']
print('Validation Loss:',val_loss)
# Code to retrieve and display key training and validation metrics ends.......


# Code to generate and display a set of three plots that
# visualize graphical representation the training and validation metrics of a neural network model!

# Code for graphical visualization starts.......
epochs = range(len(get_ac))

plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
plt.plot(epochs, get_los, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, get_ac, 'm', label='Accuracy of Training Data')
plt.plot(epochs, val_acc, 'b', label='Accuracy of Validation Data')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, get_los, 'c', label='Loss of Training Data')
plt.plot(epochs, val_loss, 'k', label='Loss of Validation Data')
plt.title('Training and Validation Loss')
plt.legend(loc=0)
plt.figure()

# To show the plot..
plt.show()

# Code for graphical visualization ends.......


# Code to evaluate the model for dataset!

# Code for evaluation of model starts.......
loss, accuracy = model.evaluate(testing_dataset)
loss, accuracy = model.evaluate(training_dataset)
loss, accuracy = model.evaluate(validation_dataset)
# Code for evaluation of model ends.......


# Code for visualizing the predictions of a neural network model 
# on a sample batch of test images from a test dataset!

# Code for visualizing the predictions starts.....
plt.subplots(figsize=(20, 20))
for images, labels in testing_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(tf.expand_dims(images[i], 0))
        score = tf.nn.softmax(predictions[0])
        if(class_names[labels[i]]==class_names[np.argmax(score)]):
            plt.title("Actual: "+class_names[labels[i]])
            plt.ylabel("Predicted: "+class_names[np.argmax(score)],fontdict={'color':'green'})
            
        else:
            plt.title("Actual: "+class_names[labels[i]])
            plt.ylabel("Predicted: "+class_names[np.argmax(score)],fontdict={'color':'red'})
        plt.gca().axes.yaxis.set_ticklabels([])        
        plt.gca().axes.xaxis.set_ticklabels([])

# Code for visualizing the predictions ends.....
        

# For printing the label and predictions array to see if they are in the proper format or not!
for img, label in testing_dataset.take(1):
    print(label, len(label))
    break

for img, label in training_dataset.take(1):
    print(label, len(label))
    break

for img, label in validation_dataset.take(1):
    print(label, len(label))
    break


# Code to examine the classification performance of the neural network model 
# on the test dataset and detailed metrics provided through the 'classification_report' function.

# Code for classification performance of the model starts........
from sklearn.metrics import classification_report,confusion_matrix

actual_label = []
pred_label = []

for img, label in testing_dataset.take(1):
#     pred = model.predict(tf.expand_dims(img, 0))
    pred = model.predict(img)
    pred = np.argmax(pred, axis=1)

#     pred_label.append(pred[0])
#     actual_label.append(label[i])


#     print(actual_label, pred_label)
    print(classification_report(label,pred))
        
# y_test_new = np.argmax(y_test,axis=1)
# Code for classification performance of the model ends........


# Code to create a heatmap of the confusion matrix to visualize the performance of a classification model!

# Code for creation of Heatmap of the confusion matrix starts........
label = np.array(label)  # Convert to NumPy array if not already
pred = np.array(pred)    # Convert to NumPy array if not already

# Find indices of empty labels or missing values
empty_label_indices = np.where(label == '')

# Remove empty labels and corresponding predictions
label = np.delete(label, empty_label_indices)
pred = np.delete(pred, empty_label_indices)

# Define your color palette
colors_green = LinearSegmentedColormap.from_list("custom_colormap", [(0, "white"), (1, "green")])
line_color = "#0000FF"

# Create confusion matrix
conf_matrix = confusion_matrix(label, pred)

# Get unique labels for plotting
unique_labels = np.unique(label)

# Create the heatmap
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(conf_matrix, ax=ax, xticklabels=unique_labels, yticklabels=unique_labels, annot=True,
            cmap=colors_green, alpha=0.7, linewidths=2, linecolor=line_color)

# Set axis labels
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

# Set title
ax.set_title('Heatmap of the Confusion Matrix', size=18, fontweight='bold', fontname='monospace', color=line_color, y=1.02)

# Display the heatmap
plt.show()
# Code for creation of Heatmap of the confusion matrix ends........