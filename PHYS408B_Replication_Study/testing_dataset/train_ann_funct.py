# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# define model
def train_ann(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(254, activation= 'relu', input_shape = (12,))) #input shape = X_train[0].shape
    model.add(Dense(64, activation= 'relu'))
    # model.add(Dense(4, activation= 'relu'))
    model.add(Dense(1))

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.01), loss='mse')

    #early stopping
    callback = tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss', mode = 'min')

    history = model.fit(X_train,y_train, epochs = 500,  batch_size=256, verbose=1, validation_data = (X_test,y_test), callbacks = [callback])

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_pred, y_test)
    mae = mean_absolute_error(y_pred, y_test)
    print(f"{X_train.shape[0]} samples")
    print("mse", mse)
    print("r2", r2)
    print("mae", mae)
    return model

# # Read in the dataset
# train = pd.read_csv('../train_test/train.csv', header = 0)
# test = pd.read_csv('../train_test/test.csv', header = 0)
#
# # Split the datasets
# y_train = train['M']
# X_train = train.drop(columns = ['M'])
#
# y_test = test['M']
# X_test = test.drop(columns = ['M'])
#
# train_ann(X_train, y_train, X_test, y_test)



