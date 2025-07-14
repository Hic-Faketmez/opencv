
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


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
train_path = "C:\Users\pc\Desktop\Yaz\opencv\6_Deep_Learning_for_CV\deeplearning_workspace\data\CATS_DOGS\train"
test_path = "C:\Users\pc\Desktop\Yaz\opencv\6_Deep_Learning_for_CV\deeplearning_workspace\data\CATS_DOGS\test"
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)