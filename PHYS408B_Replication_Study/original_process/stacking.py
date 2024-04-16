# import libraries
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle as pkl

# Source: https://github.com/dppant/magnetism-prediction/tree/main

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

#create and fit scaler using train data
scaler = StandardScaler()
scaler.fit(X_train[cols_to_scale])

#scale trained data
X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])

# scale test data
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# save standard scaler
pkl.dump(scaler, open('../models/scaler.pkl', 'wb'))

# define k-fold object
kfold = KFold(n_splits=10, shuffle=True, random_state=100)

# initialize empty dataframe
results = pd.DataFrame()

# early stopping
es = EarlyStopping(patience=15, monitor='val_loss', mode='min')

# initialize
output_svm = []
output_rf = []
output_ANN = []
output_xgb = []
output_knn = []
output_dt = []
target = []
for train, val in kfold.split(X_train):
    X_train_small, y_train_small = X_train.iloc[train], y_train.iloc[train]
    X_val, y_val = X_train.iloc[val], y_train.iloc[val]

    # model-1
    svr = SVR(C=12)
    svr.fit(X_train_small, y_train_small)
    y_pred_svm = svr.predict(X_val)

    # model-2
    rf_reg = RandomForestRegressor(n_estimators=500, min_samples_split=13)
    rf_reg.fit(X_train_small, y_train_small)
    y_pred_rf = rf_reg.predict(X_val)

    # model-3
    knn_reg = KNeighborsRegressor()
    knn_reg.fit(X_train_small, y_train_small)
    y_pred_knn = knn_reg.predict(X_val)

    # model-4
    xgb_reg = GradientBoostingRegressor()
    xgb_reg.fit(X_train_small, y_train_small)
    y_pred_xgb = xgb_reg.predict(X_val)

    # model-5
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train_small, y_train_small)
    y_pred_dt = dt_reg.predict(X_val)

    # model-6
    # define model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(12,)))  # input shape = X_train[0].shape
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # early stopping
    callback = EarlyStopping(patience=7, monitor='val_loss', mode='min')
    history = model.fit(X_train_small, y_train_small, epochs=500, batch_size=256, verbose=0,
                        validation_data=(X_val, y_val), callbacks=[es])
    y_pred_ann = model.predict(X_val)
    y_pred_ann = y_pred_ann.reshape(y_pred_ann.shape[0], )

    # append results
    output_svm.extend(y_pred_svm)
    output_rf.extend(y_pred_rf)
    output_ANN.extend(y_pred_ann)
    output_xgb.extend(y_pred_xgb)
    output_knn.extend(y_pred_knn)
    output_dt.extend(y_pred_dt)
    target.extend(y_val)

# combine predictions
results['svm'] = output_svm
results['rf'] = output_rf
results['ann'] = output_ANN
results['knn'] = output_knn
results['xgb'] = output_xgb
results['dt'] = output_dt
results['target'] = target

X_train_stacking = results[['rf','svm', 'knn', 'ann', 'dt', 'xgb']]
y_train_stacking = results['target']

# Load and prepare models
ANN_model = load_model("../models/ann.h5")

svr = SVR(C=12)
svr.fit(X_train,y_train)

rf_reg = RandomForestRegressor(n_estimators=500, min_samples_split = 13)
rf_reg.fit(X_train,y_train)

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train,y_train)

xgb_reg = GradientBoostingRegressor()
xgb_reg.fit(X_train, y_train)

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)

# save trained models using pickle
pkl.dump(svr, open('../models/svr.pkl', 'wb'))
pkl.dump(rf_reg, open('../models/rf_reg.pkl', 'wb'))
pkl.dump(knn_reg, open('../models/knn_reg.pkl', 'wb'))
pkl.dump(xgb_reg, open('../models/xgb_reg.pkl', 'wb'))
pkl.dump(dt_reg, open('../models/dt_reg.pkl', 'wb'))

# load saved models
svr = pkl.load(open('../models/svr.pkl', 'rb'))
rf_reg = pkl.load(open('../models/rf_reg.pkl', 'rb'))
knn_reg = pkl.load(open('../models/knn_reg.pkl', 'rb'))
xgb_reg = pkl.load(open('../models/xgb_reg.pkl', 'rb'))
dt_reg = pkl.load(open('../models/dt_reg.pkl', 'rb'))

# prepare new features based on base classifiers
y_pred_train_ann = ANN_model.predict(X_train)
y_pred_train_ann = y_pred_train_ann.reshape(y_pred_train_ann.shape[0],)
y_pred_train_svm = svr.predict(X_train)
y_pred_train_rf = rf_reg.predict(X_train)
y_pred_train_knn = knn_reg.predict(X_train)
y_pred_train_xgb = xgb_reg.predict(X_train)
y_pred_train_dt = dt_reg.predict(X_train)

features = pd.DataFrame()
features['ann'] = y_pred_train_ann
features['svm'] = y_pred_train_svm
features['rf'] = y_pred_train_rf
features['knn'] = y_pred_train_knn
features['xgb'] = y_pred_train_xgb
features['dt'] = y_pred_train_dt
features['target'] = np.array(y_train)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X_train_stacking = features.iloc[:,:6]
y_train_stacking = features['target']

# train final meta classifier # RF
rf_meta_final = RandomForestRegressor()
rf_meta_final.fit(X_train_stacking, y_train_stacking)

# save final meta model and load
pkl.dump(rf_meta_final, open('../models/rf_meta_final.pkl', 'wb'))
rf_meta_final = pkl.load(open('../models/rf_meta_final.pkl', 'rb'))

# prepare X-Test
# prepare new features based on base classifiers for test set
y_pred_test_ann = ANN_model.predict(X_test)
y_pred_test_ann = y_pred_test_ann.reshape(y_pred_test_ann.shape[0],)
y_pred_test_svm = svr.predict(X_test)
y_pred_test_rf = rf_reg.predict(X_test)
y_pred_test_knn = knn_reg.predict(X_test)
y_pred_test_xgb = xgb_reg.predict(X_test)
y_pred_test_dt = dt_reg.predict(X_test)

# create test data new features
test_features = pd.DataFrame()
test_features['ann'] = y_pred_test_ann
test_features['svm'] = y_pred_test_svm
test_features['rf'] = y_pred_test_rf
test_features['knn'] = y_pred_test_knn
test_features['xgb'] = y_pred_test_xgb
test_features['dt'] = y_pred_test_dt
test_features['target'] = np.array(y_test)

X_test_stacking = test_features.iloc[:,:6]
y_test_stacking = test_features['target']

# evaluate results
final_prediction = rf_meta_final.predict(X_test_stacking)
mse = mean_squared_error(final_prediction, y_test_stacking)
r2 = r2_score(final_prediction, y_test_stacking)
mae = mean_absolute_error(final_prediction, y_test_stacking)
print(mse, mae, r2)

pd.DataFrame(final_prediction).to_csv("test_predictions.csv")
pd.DataFrame(y_test_stacking).to_csv("y_test.csv")


