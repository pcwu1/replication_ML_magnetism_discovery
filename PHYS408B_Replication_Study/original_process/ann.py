import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#import libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR
import sklearn
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

############################################################################################################

# read csv
train = pd.read_csv('../train_test/train.csv', header = 0)
test = pd.read_csv('../train_test/test.csv', header = 0)

# split X and y
y_train = train['M']
X_train = train.drop(columns = ['M'])

y_test = test['M']
X_test = test.drop(columns = ['M'])

# data normalization
from sklearn.preprocessing import StandardScaler

# list for cols to scale
cols_to_scale = ['Fe','S1','S2','S3','S4','Ni','Co','Cr','Mn','Se','S','Te']

############################################################################################################

#create and fit scaler using train data
scaler = StandardScaler()
scaler.fit(X_train[cols_to_scale])

#scale trained data
X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])

# scale test data
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

############################################################################################################

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=100)

sum_mse, sum_r2, sum_mae = 0, 0, 0
for train, val in kfold.split(X_train):
    X_train_small, y_train_small = X_train.iloc[train], y_train.iloc[train]
    X_val, y_val = X_train.iloc[val], y_train.iloc[val]

    # define model
    model = Sequential()
    model.add(Dense(254, activation='relu', input_shape=(12,)))  # input shape = X_train[0].shape
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(4, activation= 'relu'))
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

    # early stopping
    callback = tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss', mode='min')

    history = model.fit(X_train_small, y_train_small, epochs=500, batch_size=256, verbose=1,
                        validation_data=(X_val, y_val), callbacks=[callback])

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_pred, y_val)
    r2 = r2_score(y_pred, y_val)
    mae = mean_absolute_error(y_pred, y_val)
    sum_mse = sum_mse + mse
    sum_r2 = sum_r2 + r2
    sum_mae = sum_mae + mae
    print(str(mse), str(r2), str(mae))

print("-----------------------")
print("Average = ", str(sum_mse / 10), str(sum_r2 / 10), str(sum_mae / 10))
print("-----------------------")
# Average =  1.4545837994777284 0.9344226687752268 0.5838360607494378

import keras
from matplotlib import pyplot as plt
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
# plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig("../images/ann_graph.png")
plt.show()

model.save('../models/ann.h5')