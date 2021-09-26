import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

from variables import*

def move_files(source, destination):
    shutil.move(source, destination)

############################################################################################

def make_class_dir(): # This Only Runs OneTime
    df_labels = pd.read_csv(train_labels)
    df_labels['label'] = df_labels['label'].astype('int')
    train_images = os.listdir(train_dir)

    image_ids = df_labels['image_id'].values
    labels = df_labels['label'].values
    for label, image_id in zip(labels, image_ids):
        if image_id in train_images:
            label = 0 if (label == 4) else 1
            class_dir = os.path.join(train_dir, str(label))
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
            move_files(os.path.join(train_dir, image_id), class_dir)

############################################################################################

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def image_data_generator():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                    rotation_range = rotation_range,
                                                    shear_range = shear_range,
                                                    zoom_range = zoom_range,
                                                    width_shift_range=shift_range,
                                                    height_shift_range=shift_range,
                                                    horizontal_flip = True,
                                                    validation_split= val_split,
                                                    preprocessing_function=preprocessing_function
                                                                )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    class_mode='binary',
                                    subset = 'training',
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    class_mode = 'binary',
                                    subset = 'validation',
                                    shuffle = True)

    return train_generator, validation_generator