from math import trunc
import random
import numpy as np
import pandas as pd
import pickle as pkl
from tensorflow.keras.models import load_model
import itertools

# Source: https://github.com/dppant/magnetism-prediction/tree/main
def findMatchings(X):
    A = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
    n = len(A)
    combinations = []
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            l = j + 1
            r = n - 1
            while l < r:
                if A[i] + A[j] + A[l] + A[r] == X:
                    if [A[i], A[j], A[l], A[r]] not in combinations:
                        combinations.append([A[i], A[j], A[l], A[r]])
                    l += 1
                    r -= 1
                elif A[i] + A[j] + A[l] + A[r] < X:
                    l += 1
                else:
                    r -= 1

  # calculate permutations of each found combinations
  # print(combinations)

    permutations = []
    for item in combinations:
        temp = list(itertools.permutations(item))
        t = [list(e) for e in temp]
        permutations.extend(t)

    return list(map(list, set(map(tuple, permutations))))

def generate_new_input_and_predict():
    '''
    This functions creates a bimetallic chalcogenides by randomly choosing a transition metal, its comcentration
    and chalcogen element.
    '''
    trans_metals = ['Ni', 'Co', 'Cr', 'Mn']
    chalcogens = ['S', 'Se', 'Te']

    y_all = np.linspace(6, 62.5, 1131)

    for A in trans_metals:
        for B in chalcogens:
            for y in y_all:
                x = 100 - y  # percentage of Fe
                n = round(16 * y / 100)  # Total number of substituted transition metalfor given substitution
                # call function to find all possible permutation of four integers that sums to n
                S_all = findMatchings(n)

                for S_item in S_all:
                    S1, S2, S3, S4 = S_item
                    #                     return [A,B,x,y,S1,S2,S3,S4]
                    sample = [A, B, x / 100, y / 100, S1, S2, S3, S4]

                    # create new dataframe
                    cols = ['Fe', 'S1', 'S2', 'S3', 'S4', 'Ni', 'Co', 'Cr', 'Mn', 'Se', 'S',
                            'Te']  # list for cols to scale
                    df_new = pd.DataFrame(columns=cols)
                    data = [0] * 12
                    df_new.loc[len(df_new)] = data

                    transition_metal = sample[0]
                    chalcogen = sample[1]

                    df_new['Fe'] = sample[2]
                    df_new[transition_metal] = sample[3]
                    df_new[chalcogen] = 1
                    df_new['S1'] = sample[4]
                    df_new['S2'] = sample[5]
                    df_new['S3'] = sample[6]
                    df_new['S4'] = sample[7]

                    X_test = df_new.copy()

                    # load standard scaler
                    scaler = pkl.load(open('../models/scaler.pkl', 'rb'))

                    # transform input data using saved standard scaler
                    cols_to_scale = ['Fe', 'S1', 'S2', 'S3', 'S4', 'Ni', 'Co', 'Cr', 'Mn', 'Se', 'S',
                                     'Te']  # list for cols to scale
                    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])  # scale test data

                    # load base models
                    ANN_model = load_model("../models/ann.h5")
                    svr = pkl.load(open('../models/svr.pkl', 'rb'))
                    rf_reg = pkl.load(open('../models/rf_reg.pkl', 'rb'))
                    knn_reg = pkl.load(open('../models/knn_reg.pkl', 'rb'))
                    xgb_reg = pkl.load(open('../models/xgb_reg.pkl', 'rb'))
                    dt_reg = pkl.load(open('../models/dt_reg.pkl', 'rb'))

                    # load meta model
                    rf_meta_final = pkl.load(open('../models/rf_meta_final.pkl', 'rb'))

                    # prepare new features based on base classifiers for test set
                    y_pred_test_ann = ANN_model.predict(X_test, verbose=0)
                    y_pred_test_ann = y_pred_test_ann.reshape(y_pred_test_ann.shape[0], )
                    y_pred_test_svm = svr.predict(X_test)
                    y_pred_test_rf = rf_reg.predict(X_test)
                    y_pred_test_knn = knn_reg.predict(X_test)
                    y_pred_test_xgb = xgb_reg.predict(X_test)
                    y_pred_test_dt = dt_reg.predict(X_test)

                    # get new features from base models
                    X_test_stacking = pd.DataFrame()
                    X_test_stacking['ann'] = y_pred_test_ann
                    X_test_stacking['svm'] = y_pred_test_svm
                    X_test_stacking['rf'] = y_pred_test_rf
                    X_test_stacking['knn'] = y_pred_test_knn
                    X_test_stacking['xgb'] = y_pred_test_xgb
                    X_test_stacking['dt'] = y_pred_test_dt

                    # apply new features to meta classifier
                    final_prediction = rf_meta_final.predict(X_test_stacking)

                    # write results to file
                    df_new['M_predicted'] = final_prediction
                    df_new.to_csv("exhaustive_search_results.csv", mode='a+', index=False, header=False)

generate_new_input_and_predict()