# this file contains the python file for data preprocessing.
# Data should be organanized, reshaped, cropped and clean the dataset.
# for full process, use the 'training.py' script.
mport tensorflow as tf
import numpy as np
import os
import cv2
from tf_keras.preprocessing.image import ImageDataGenerator

# define path:
dataset_path = './dataset/'

# Image dimensions
IMG_SIZE = 224
BATCH_SIZE = 32

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Get class labels
class_labels = list(train_generator.class_indices.keys())
print("Class Labels:", class_labels)
