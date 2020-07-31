import featuretools as ft
import lightgbm as lgb
import optuna
import numpy as np
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import xgboost as xgb
import re
import seaborn as sns
from tensorflow import keras
#import keras.layers as L
import seaborn as sns
from datetime import datetime, timezone, timedelta
#from keras.models import Model
from sklearn.decomposition import PCA
#from keras import losses
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
import unicodedata


train_path = "../input/train_data.csv"
test_path = "../input/test_data.csv"

""" load raw data"""
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

""" Preprocessing"""
import preprocess as pr
import impute as im

import copy

df = train["y"]

predata = pd.concat([train.drop("y", axis=1), test], ignore_index=True)
predata_copy = copy.deepcopy(predata)
"""predata_onehot = pr.Preprocessor(predata).all("onehot")"""
predata_label = pr.Preprocessor(predata_copy).all("label", "date")

"""prep_train_onehot = pd.concat([df, predata_onehot.iloc[:len(train), :]], axis=1)
prep_test_onehot = predata_onehot.iloc[len(train):, :]"""

prep_train_label = pd.concat([df, predata_label.iloc[:len(train), :]], axis=1)
prep_test_label = predata_label.iloc[len(train):, :]

"""prep_train_onehot.to_csv("../prep_train_onehot.csv", index=False)
prep_test_onehot.to_csv("../prep_test_onehot.csv", index=False)
prep_train_label.to_csv("../prep_train_label.csv", index=False)
prep_test_label.to_csv("../prep_test_label.csv", index=False)"""

""" define data"""
train_X = prep_train_label.drop([
    "y", "video_id", "thumbnail_link", "publishedAt", "collection_date",
    "id", "tags", "description", "title"], axis=1
    )
""""publishedAt", "collection_date","""
train_y = np.log1p(prep_train_label["y"])
"""train_y = np.log(prep_train_label["y"])"""
test_X = prep_test_label.drop([
    "video_id", "thumbnail_link", "publishedAt", "collection_date",
    "id", "tags", "description", "title"], axis=1
    )
""""publishedAt",  "collection_date","""

""" target encoding"""
from feature_selection import FeatureSelector as FS, cross_validator
train_X_te, test_X_te = FS(train_X, train_y).target_encoder(test_X)



""" feature selection"""
#selected = FS(train_X, train_y).greedy_forward_selection()
selected_te = FS(train_X_te, train_y).greedy_forward_selection()
#print("selected features:"+ str(selected))
print("selected target encoding features:"+ str(selected_te))


def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'tree_method': 'hist',
        "learning_rate":trial.suggest_loguniform('learning_rate', 0.005, 0.03),
        'lambda_': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        #'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        #'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        #'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        #'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    FOLD_NUM = 4
    kf = KFold(n_splits=FOLD_NUM,
              #shuffle=True,
              random_state=42)
    scores = []
    feature_importance_df = pd.DataFrame()

    pred_cv = np.zeros(len(test.index))
    num_round = 10000


    for i, (tdx, vdx) in enumerate(kf.split(train_X_te[selected_te], train_y)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = train_X_te[selected_te].iloc[tdx], train_X_te[selected_te].iloc[vdx], train_y.values[tdx], train_y.values[vdx]
        # XGB
        xgb_dataset = xgb.DMatrix(X_train, label=y_train)
        xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
        xgbm = xgb.train(params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'),(xgb_test_dataset, 'eval')],
                          early_stopping_rounds=100, verbose_eval=5000)
        xgbm_va_pred = xgbm.predict(xgb.DMatrix(X_valid))
        xgbm_va_pred[xgbm_va_pred<0] = 0
        score_ = np.sqrt(mean_squared_error(y_valid, xgbm_va_pred))
        scores.append(score_)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 結果の確認
print('Best trial:')
light_trial = study.best_trial

print('  Value: {}'.format(light_trial.value))

print('  Params_xgb: ')

with open("lightgbmparams.txt", "w") as file:
    for key, value in light_trial.params.items():
       print('    "{}": {},'.format(key, value))
       file.write('"{}": {},'.format(key, value))