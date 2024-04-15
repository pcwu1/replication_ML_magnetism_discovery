# import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle as pkl
from imbalance_funct import imbalance_check
from train_ann_funct import train_ann
import matplotlib.pyplot as plt

# read csv
train = pd.read_csv('../train_test/train.csv', header = 0)
test = pd.read_csv('../train_test/test.csv', header = 0)

# % of the full training set of 3695 samples
partition_n = [int(train.shape[0] * 0.9),
               int(train.shape[0] * 0.8),
               int(train.shape[0] * 0.7),
               int(train.shape[0] * 0.6),
               int(train.shape[0] * 0.5),
               int(train.shape[0] * 0.4),
               int(train.shape[0] * 0.3),
               int(train.shape[0] * 0.2),
               int(train.shape[0] * 0.1)]

# Subsample from the training set
train_sets = []
for n in partition_n:
    temp_train = train.sample(n=n, replace=False, random_state=42)
    temp_train.reset_index(drop=True, inplace=True)
    train_sets.append(temp_train)

# Ensure the different sized datasets are balanced
train_sets_data = []
for set in train_sets:
    temp_data = imbalance_check(train, set, save_name=f"{set.shape[0]}_imbalance", save=True)
    print(f"{set.shape[0]} samples dataset imbalance check:")
    print(temp_data, "\n")
    train_sets_data.append(temp_data)

data_results = []
for set in train_sets:
    # Split the dataset
    y_train = set['M']
    X_train = set.drop(columns=['M'])

    y_test = test['M']
    X_test = test.drop(columns=['M'])

    ann_model = train_ann(X_train, y_train, X_test, y_test)

    svr = SVR(C=12)
    svr.fit(X_train, y_train)

    rf_reg = RandomForestRegressor(n_estimators=500, min_samples_split=13)
    rf_reg.fit(X_train, y_train)

    knn_reg = KNeighborsRegressor()
    knn_reg.fit(X_train, y_train)

    xgb_reg = GradientBoostingRegressor()
    xgb_reg.fit(X_train, y_train)

    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_train)

    # prepare new features based on base classifiers
    y_pred_train_ann = ann_model.predict(X_train)
    y_pred_train_ann = y_pred_train_ann.reshape(y_pred_train_ann.shape[0], )
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

    X_train_stacking = features.iloc[:, :6]
    y_train_stacking = features['target']

    # train final meta classifier # RF
    rf_meta_final = RandomForestRegressor()
    rf_meta_final.fit(X_train_stacking, y_train_stacking)

    # prepare X-Test
    # prepare new features based on base classifiers for test set
    y_pred_test_ann = ann_model.predict(X_test)
    y_pred_test_ann = y_pred_test_ann.reshape(y_pred_test_ann.shape[0], )
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

    X_test_stacking = test_features.iloc[:, :6]
    y_test_stacking = test_features['target']

    # evaluate results
    final_prediction = rf_meta_final.predict(X_test_stacking)
    mse = mean_squared_error(final_prediction, y_test_stacking)
    r2 = r2_score(final_prediction, y_test_stacking)
    mae = mean_absolute_error(final_prediction, y_test_stacking)

    temp_dict = {}
    temp_dict["num_samples"] = set.shape[0]
    temp_dict["mse"] = mse
    temp_dict["r2"] = r2
    temp_dict["mae"] = mae
    data_results.append(temp_dict)
    #
    # # Sort both lists by ascending order by y_test
    # y_test_sorted = y_test_stacking.sort_values()
    # final_prediction_series = pd.Series(final_prediction)
    # final_prediction_sorted = final_prediction_series.reindex(y_test_sorted.index)
    #
    # # Reset the index
    # y_test_sorted.reset_index(drop=True, inplace=True)
    # final_prediction_sorted.reset_index(drop=True, inplace=True)
    #
    # # Drop old indices
    # y_test_sorted.drop(columns=['Unnamed: 0'], inplace=True)
    # final_prediction_sorted.drop(columns=['Unnamed: 0'], inplace=True)
    #
    # fig = plt.figure(figsize=(12, 6), dpi=600)
    # ax = fig.add_subplot(111)
    #
    # ax.axvline(x=25, color='green', linestyle='--')
    # ax.axvline(x=332, color='green', linestyle='--')
    #
    # ax.set_title(f"{set.shape[0]} samples predictions")
    # ax.set_xlabel('Sample Number')
    # ax.set_ylabel('Magnetism')
    #
    # ax.plot(y_test_sorted, 'r', label="Actual Value", linewidth=0.5)
    # ax.plot(final_prediction_sorted, label="Predicted Value", linewidth=0.5)
    # ax.legend()
    #
    # fig.savefig(f"../images/output/{set.shape[0]}_output.svg", dpi=1200)

results_df = pd.DataFrame(data_results)
path = "different_partitions_data.csv"
# Write the DataFrame to a CSV file
results_df.to_csv(path, index=False)