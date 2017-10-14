# Reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.set_random_seed(1)
# 0. Loading Dataset:
time_steps = 24
dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
print(values.shape)

# Encode String Feature(wind direction) to integer
encoder = LabelEncoder()
encoder.fit(values[: ,4])
print(list(encoder.classes_))
values[: ,4] = encoder.transform(values[: ,4])

# ensure all data is float
values = values.astype('float32')
# normalize each feature individually
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)
n_train_hours = 365 * 24
train = values[:n_train_hours, :].reshape(-1, 1, 8)
test = values[n_train_hours:, :].reshape(-1, 1, 8)

train_samples = len(train)-time_steps
test_samples = len(test)-time_steps
train_data = np.zeros((train_samples, time_steps, 8))
test_data = np.zeros((test_samples, time_steps, 8))
for i in range(time_steps):
    train_data[:,i,:] = train[i:(train_samples+i),0,:]
    test_data[:,i,:] = test[i:(test_samples+i),0,:]
# split into input and outputs
train_X, train_y = train_data[:-1,:, 1:], train[time_steps+1:,:, 0]
test_X, test_y = test_data[:-1,:, 1:], test[time_steps+1:,:, 0]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(units=50,
               return_sequences=True,
               input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# make a prediction
yhat = model.predict(test_X)
test_X = np.squeeze(test_X[:,0,:])
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[: ,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[: ,0]
# calculate RMSE
mse = mean_squared_error(inv_y, inv_yhat)
print('Test MSE: %.3f' % mse)

# plot
plt.plot(inv_y, label='real')
plt.plot(inv_yhat, label='predicted')
plt.legend()
plt.show()