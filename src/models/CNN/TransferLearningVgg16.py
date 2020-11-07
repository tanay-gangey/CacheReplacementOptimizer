# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:33:26 2020

@author: Pranav Aditya
"""

import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


filename = "cheetah.cs.fiu.edu-110108-113008.2.blkparse"
#filename = "cheetah.1000"



df = pd.read_csv(filename, sep=' ',header = None)
df.columns = ['timestamp','pid','pname','blockNo', 'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']
print(df.head())
df1=df.head()

df2=df[:44719]
scaler = MinMaxScaler(feature_range=(0, 1024), copy=True)
scaler.fit(df2['blockNo'].values.reshape(-1,1))
tdf=scaler.transform(df2['blockNo'].values.reshape(-1,1))

X, Y = list(), list()
for i in range(len(tdf)):
		# find the end of this pattern
        end_ix = i + 32
		# check if we are beyond the sequence
        if end_ix > len(tdf)-1:
            break
		# gather input and output parts of the pattern
        seq_x, seq_y = tdf[i:end_ix], tdf[end_ix]
        arr=[]
        for i in range(32):
            arr.append(seq_x)
        ar1=copy.deepcopy(arr)
        s=[]
        s.append(ar1)
        s.append(ar1)
        s.append(ar1)
        s1=copy.deepcopy(s)
        
        
        X.append(np.array(s1))
        Y.append(seq_y)
X=np.array(X)
Y=np.array(Y)


X = X.reshape((X.shape[0], X.shape[2],X.shape[3],X.shape[1]))
X=X.astype('float32')
df3=df[44769:89488]
scaler.fit(df3['blockNo'].values.reshape(-1,1))
tdf1=scaler.transform(df3['blockNo'].values.reshape(-1,1))

X_test, Y_test = list(), list()
for i in range(len(tdf1)):
		# find the end of this pattern
        end_ix = i + 32
		# check if we are beyond the sequence
        if end_ix > len(tdf)-1:
            break
		# gather input and output parts of the pattern
        seq_x, seq_y = tdf[i:end_ix], tdf[end_ix]
        arr=[]
        for i in range(32):
            arr.append(seq_x)
        ar1=copy.deepcopy(arr)
        s=[]
        s.append(ar1)
        s.append(ar1)
        s.append(ar1)
        s1=copy.deepcopy(s)
        
        
        X_test.append(np.array(s1))
        Y_test.append(seq_y)
X_test=np.array(X_test)
Y_test=np.array(Y_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2],X_test.shape[3],X_test.shape[1]))


base_model = tf.keras.applications.VGG16(input_shape=(32,32,3),
                                               include_top=False,
                                               weights='imagenet')


feature_batch = base_model(X)
print(feature_batch.shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
pred1=tf.keras.layers.Dense(64, activation='relu')
pred1batch = pred1(feature_batch_average)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(pred1batch)
print(prediction_batch)
base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  pred1,
  prediction_layer
])

base_learning_rate = 0.0005
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.MSE,
              metrics=['accuracy'])


history = model.fit(X,Y,
                    epochs=15,
                    verbose=1)

predictions = model.predict(X_test,verbose=1)


x=list(range(0,10000))
x=np.array(x)
x=np.reshape(x, (10000, 1))
plt.plot(x, Y_test[30000:40000], label = "atual") 
plt.plot(x, predictions[30000:40000], label = "predicted") 

plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('block number') 
# giving a title to my graph 
plt.title('Actual VS Predicted') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 

 
