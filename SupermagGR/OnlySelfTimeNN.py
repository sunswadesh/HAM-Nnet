# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:18:09 2020

@author: Swadesh
"""
## A simple model to predict a noisy sinusoid varying with time
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 16, 10

csv_path = 'Magnetometers/Supermag/20201208-17-20-supermag.csv'

df = pd.read_csv(csv_path)

df = df[['dbe_nez']]


plt.figure()
plt.plot(df)
train_size = int(len(df) * 0.3)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
        # ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.dbe_nez, time_steps)
X_test, y_test = create_dataset(test, test.dbe_nez, time_steps)
xtot, ytot = create_dataset(df, df.dbe_nez, time_steps)
# X_train, y_train = create_dataset(train, train.index, time_steps)
# X_test, y_test = create_dataset(test, test.index, time_steps)

print(X_train.shape, y_train.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(
  units=32,
  input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(tf.keras.layers.Dense(units=1))
model.compile(
  loss='mean_squared_error',
  optimizer=tf.keras.optimizers.Adam(0.001)
)

history = model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=160,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)


ytn = model.predict(X_train) 
plt.figure()
plt.plot(y_train)
plt.plot(ytn)

# y_pred = model.predict(X_test)
# plt.figure()
# plt.plot(y_test)
# plt.plot(y_pred)

# ytot_pred = model.predict(xtot)
# ysum = np.append(ytn,y_pred)
# plt.figure()
# plt.plot(ytot)
# plt.plot(ytot_pred)
# plt.plot(ysum)



## Train on the whole dataset. This should be equivalent to an empirical model.
# history = model.fit(
#     xtot, ytot,
#     epochs=10,
#     batch_size=160,
#     validation_split=0.1,
#     verbose=1,
#     shuffle=False
# )
# ytot_pred = model.predict(xtot)
# plt.figure()
# plt.plot(ytot)
# plt.plot(ytot_pred)


