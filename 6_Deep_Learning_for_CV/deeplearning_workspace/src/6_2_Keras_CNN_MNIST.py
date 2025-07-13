import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from keras import saving

# Load MNIST data set

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Encode Y_test labels
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)

# Normalize the Data
x_train = x_train/255
x_test = x_test/255

# Reshape the Data

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) # 60000, 28, 28, 1
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # 10000, 28, 28, 1

# Train the Model

model = Sequential()

# One Set, as our model is simple we will use one set only
# Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28, 1), activation='relu'))

# Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))

# Flattening the image from 28x28 to 784 before the final layer
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
# Multiclassfication problem => softmax
model.add(Dense(10, activation='softmax'))

# https://keras.io/metrics/    => can refer various metrics avaliable
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

print(model.summary())

# Add Early Stopping

early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1) 

# Fit the Model

model.fit(x_train, y_cat_train, 
          validation_data=(x_test, y_cat_test),
          epochs=10, 
          callbacks=[early_stopping])

# Evaluate the Model

metrics = pd.DataFrame(model.history.history)
print(metrics.head())

loss_metrics = metrics[['loss', 'val_loss']].plot()
plt.show(loss_metrics)

acc_metrics = metrics[['accuracy', 'val_accuracy']].plot()
plt.show(acc_metrics)

print("model evaluation:", model.evaluate(x_test, y_cat_test, verbose=0))

predictions = np.argmax(model.predict(x_test), axis=-1)

matrix = confusion_matrix(y_test, predictions)
print("confusion matrix:", matrix)

report = classification_report(y_test, predictions)
print("classification report:", report)

saving.save_model(model, 'models/cnn_mnist.keras')
