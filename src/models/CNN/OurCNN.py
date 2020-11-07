# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:05:03 2020

@author: Pranav Aditya
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

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
	end_ix = i + 30
		# check if we are beyond the sequence
	if end_ix > len(tdf)-1:
		break
		# gather input and output parts of the pattern
	seq_x, seq_y = tdf[i:end_ix], tdf[end_ix]
	X.append(np.array(seq_x))
	Y.append(seq_y)
X=np.array(X)
Y=np.array(Y)


X = X.reshape((X.shape[0], X.shape[1], 1))

df3=df[44769:89488]
scaler.fit(df3['blockNo'].values.reshape(-1,1))
tdf1=scaler.transform(df3['blockNo'].values.reshape(-1,1))

X_test, Y_test = list(), list()
for j in range(len(tdf1)):
		# find the end of this pattern
        end_jx = j + 30
		# check if we are beyond the sequence
        if (end_jx > len(tdf1)-1):
            break
		# gather input and output parts of the pattern
        seq_xx= tdf1[j:end_jx]
        seq_yy=tdf1[end_jx]
        X_test.append(np.array(seq_xx))
        Y_test.append(seq_yy)
X_test=np.array(X_test)
Y_test=np.array(Y_test)




model = models.Sequential()
model.add(layers.Conv1D(64,2, activation='relu', input_shape=(30,1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


history = model.fit(X, Y, epochs=100, 
                    validation_data=(X_test, Y_test))

loss, acc = model.evaluate(X_test, Y_test, verbose=1)
predictions = model.predict(X_test)
x=list(range(1,500+1))
x=np.array(x)
x=np.reshape(x, (500, 1))
plt.plot(x, Y_test[0:500], label = "line 1") 
plt.plot(x, predictions[0:500], label = "line 2") 

plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('Two lines on same graph!') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 


