import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from keras import saving

# Load MNIST data set

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Encode Y_test labels
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)

# Normalize the Data
x_train = x_train/255
x_test = x_test/255

# Train the Model

model = Sequential()

# 2 Convolutional Layers
# Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32, 32, 3), activation='relu'))
# Pooling Layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(4,4), input_shape=(32, 32, 3), activation='relu'))
# Pooling Layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Flatten the layers
model.add(Flatten())

model.add(Dense(256, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Add Early Stopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1) 

# Fit the Model

model.fit(x_train, y_cat_train, 
          validation_data=(x_test, y_cat_test),
          epochs=15, 
          callbacks=[early_stopping],
          verbose=1)

# Evaluate the Model

metrics = pd.DataFrame(model.history.history)
print(metrics.head())

loss_metrics = metrics[['loss', 'val_loss']].plot()
plt.show()

acc_metrics = metrics[['accuracy', 'val_accuracy']].plot()
plt.show()

print("model evaluation:", model.evaluate(x_test, y_cat_test, verbose=0))

predictions = np.argmax(model.predict(x_test), axis=-1)

matrix = confusion_matrix(y_test, predictions)
print("confusion matrix:", matrix)

report = classification_report(y_test, predictions)
print("classification report:", report)

saving.save_model(model, 'models/cnn_cifar10.keras')
