# Keras Basics

import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras import load_model

data_path = 'bank_note_data.txt'

data=genfromtxt(data_path, delimiter=',')
labels=data[:,4]
features=data[:,0:4]
X=features
y=labels 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

scalerObject=MinMaxScaler()
scalerObject.fit(X_train)
scaled_X_train=scalerObject.transform(X_train)
scaled_X_test=scalerObject.transform(X_test)
    
model=Sequential()
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
    
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
model.fit(scaled_X_train,y_train,epochs=50,verbose=2)
print(model.fit(scaled_X_train,y_train,epochs=50,verbose=2))

# model.predict(scaled_X_test)

predictions = model.predict_classes(scaled_X_test)

matrix = confusion_matrix(y_test, predictions)
print (matrix)

report = classification_report(y_test, predictions)
print(report)

model.save('my_model.h5')

new_model = load_model('my_model.h5')