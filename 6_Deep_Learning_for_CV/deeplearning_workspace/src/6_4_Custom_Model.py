
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras. callbacks import EarlyStopping


image_gen = ImageDataGenerator(rotation_range=25, # rotate the image 25 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 10%
                               height_shift_range=0.10, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
train_path = "C:\Users\pc\Desktop\Yaz\opencv\6_Deep_Learning_for_CV\deeplearning_workspace\data\CATS_DOGS\train"
test_path = "C:\Users\pc\Desktop\Yaz\opencv\6_Deep_Learning_for_CV\deeplearning_workspace\data\CATS_DOGS\test"
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

image_shape = (150, 150, 3)

model = Sequential()

# Convoluntional Layers
model.add(Conv2D(filters=32, 
                 kernel_size=(3,3), 
                 input_shape=image_shape, #image_shape = (130, 130, 3)
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, 
                 kernel_size=(3,3), 
                 input_shape=image_shape,
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, 
                 kernel_size=(3,3), 
                 input_shape=image_shape, 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# flatten out the layers
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))

# Dropout layer
# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Output lyaer
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()

