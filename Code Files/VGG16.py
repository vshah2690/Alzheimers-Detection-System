# Import the required libraries
import os
import numpy as np
import warnings
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

print('Libraries Imported')

data_directory = '../Alzheimer_Dataset/WHOLE_DATASET'

# Splitting the Dataset into training, validation, and testing
train_split = 0.8
val_split = 0.1
test_split = 0.1

IMG_HEIGHT = 128
IMG_WIDTH = 128

train_dataset = image_dataset_from_directory(
    data_directory,
    labels='inferred',
    label_mode='categorical',
    validation_split=val_split + test_split,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
)

val_dataset = image_dataset_from_directory(
    data_directory,
    labels='inferred',
    label_mode='categorical',
    validation_split=val_split + test_split,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
)

test_dataset = image_dataset_from_directory(
    data_directory,
    labels='inferred',
    label_mode='categorical',
    validation_split=val_split + test_split,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
)

print('Split Done')

# Define the VGG16 model
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)
base_model.trainable = False

print('Base Model Declared')

# Add custom layers for classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
ModelTraining = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
)

print('Started Training')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)

print(test_accuracy)