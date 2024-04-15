# import libraries
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential, load_model
import pickle as pkl
import tensorflow as tf

# To ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Load and prepare models
ANN_model = load_model("../models/ann.h5")

svr = SVR(C=12)
svr.fit(X_train,y_train)

rf_reg = RandomForestRegressor(n_estimators=500, min_samples_split = 13, random_state=42)
rf_reg.fit(X_train,y_train)

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train,y_train)

xgb_reg = GradientBoostingRegressor(random_state=42)
xgb_reg.fit(X_train, y_train)

dt_reg = DecisionTreeRegressor(random_state=42)
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
ANN_model = load_model("../models/ann.h5")

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

# Generate column combinations
combinations_list = []
learners = features[['rf','svm', 'knn', 'ann', 'dt', 'xgb']]
for num in range(1, 7):
    column_combinations = list(combinations(learners.columns, num))
    combinations_list.extend(column_combinations)

# combinations_list = combinations_list[-2:]
print(combinations_list)

# train and test target values
y_train_stacking = features['target']
y_test_stacking = test_features['target']

# Prepare lists to store results
pred_results_list = []
cv_results_list = []
cv_mean_list = []

# Loop through combinations
for combination in combinations_list:
    X_train_stacking = np.empty((len(features), 0))
    X_test_stacking = np.empty((len(test_features), 0))

    # Stack the features from the selected models
    for model in combination:
        X_train_stacking = np.hstack((X_train_stacking, features[model].values.reshape(-1, 1)))
        X_test_stacking = np.hstack((X_test_stacking, test_features[model].values.reshape(-1, 1)))

    # Compute the CV scores on the training set for the combination
    rf_meta_final = RandomForestRegressor(random_state=42)
    mse_cv_score = cross_val_score(rf_meta_final, X_train_stacking, y_train_stacking, cv= 10, scoring='neg_mean_squared_error')
    mae_cv_score = cross_val_score(rf_meta_final, X_train_stacking, y_train_stacking, cv=10, scoring='neg_mean_absolute_error')
    r2_cv_score = cross_val_score(rf_meta_final, X_train_stacking, y_train_stacking, cv=10)

    # Save the scores and means
    cv_results_list.append([combination, mse_cv_score, mae_cv_score, r2_cv_score])
    cv_mean_list.append([combination, mse_cv_score.mean(), mae_cv_score.mean(), r2_cv_score.mean()])

    # Train the model
    rf_meta_final.fit(X_train_stacking, y_train_stacking)

    # save final meta model and load
    pkl.dump(rf_meta_final, open(f'../comb_models/{combination}.pkl', 'wb'))
    rf_meta_final = pkl.load(open(f'../comb_models/{combination}.pkl', 'rb'))

    # evaluate results on the test set
    final_prediction = rf_meta_final.predict(X_test_stacking)
    mse = mean_squared_error(final_prediction, y_test_stacking)
    r2 = r2_score(final_prediction, y_test_stacking)
    mae = mean_absolute_error(final_prediction, y_test_stacking)
    pred_results_list.append([combination, mse, mae, r2])

    # Save the predictions of the combination of models
    pd.DataFrame(final_prediction).to_csv(f"../comb_csv_files/{combination}_test_predictions.csv")

# Save the results in csv files
pd.DataFrame(y_test_stacking).to_csv("y_test.csv")

cv_results_df = pd.DataFrame(cv_results_list, columns=['Combination', 'MSE', 'MAE', 'R2'])
cv_results_df.to_csv('cv_combination_scores.csv', index=False)

cv_means_df = pd.DataFrame(cv_mean_list, columns=['Combination', 'MSE', 'MAE', 'R2'])
cv_means_df.to_csv('cv_combination_mean_scores.csv', index=False)

pred_results_df = pd.DataFrame(pred_results_list, columns=['Combination', 'MSE', 'MAE', 'R2'])
pred_results_df.to_csv('pred_combination_scores.csv', index=False)