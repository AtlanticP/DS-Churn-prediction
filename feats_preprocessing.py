import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from ohe_topN import OHE_topN
  
import pickle as pkl  
import pandas as pd
from tqdm import tqdm
# import shap 
import matplotlib.pyplot as plt

import numpy as np 
# import math
#%%
SEED = 32
with open('pkl/df_filled_train.pkl', 'rb') as file:
    DATA = pkl.load(file)
    
df_train = DATA['df_train']
feats_num = DATA['feats_num']
feats_cat = DATA['feats_cat']

with open('pkl/df_filled_valid.pkl', 'rb') as file:
    DATA = pkl.load(file)
df_valid = DATA['df_valid']

with open('pkl/X_filled_test.pkl', 'rb') as file:
    DATA = pkl.load(file)
X_test = DATA['X_test']

print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of X_test:', X_test.shape)

X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_valid = df_valid.iloc[:, :-1]
y_valid = df_valid.iloc[:, -1]
#%% Numerical features

print('Transformation of numerical features. Standard scaling.')
# numerical features
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_train[feats_num])
X_num_train = pd.DataFrame(X_num_train, columns=feats_num, index=X_train.index)
X_num_valid = scaler.transform(X_valid[feats_num])
X_num_valid = pd.DataFrame(X_num_valid, columns=feats_num, index=X_valid.index)
X_num_test = scaler.fit_transform(X_test[feats_num])
X_num_test = pd.DataFrame(X_num_test, columns=feats_num)
#%% Categorical features
encoder = OHE_topN(top_n=10)
# encoder = LabelEncoder()
print(f'Transformation of categorical features. {encoder.__class__.__name__}.')
#%% OHE_top10 encoder
encoder.fit(X_train[feats_cat])
X_cat_train = encoder.transform(X_train[feats_cat])
X_cat_valid = encoder.transform(X_valid[feats_cat])
X_cat_test = encoder.transform(X_test[feats_cat])
feats_cat = X_cat_train.columns
#%% concatenatenate numerical and categorical data

X_clean_train = pd.concat([X_num_train, X_cat_train], axis=1)
X_clean_valid = pd.concat([X_num_valid, X_cat_valid], axis=1)
X_clean_test = pd.concat([X_num_test, X_cat_test], axis=1)
feats = X_clean_train.columns.tolist()
#%% functions to fit a model

def roc_auc_feval(preds, train_data):
    ''' 
    Each evaluation function should accept two parameters: 
    preds, train_data, and 
    return (eval_name, eval_result, is_higher_better) 
    or list of such tuples
    '''
    score = roc_auc_score(y_valid, preds)
    return 'roc_auc_score', score, True 

def fit(X_train, y_train, X_valid=X_clean_valid, y_valid=y_valid, params=None):
    
    # undersampling data
    rus = RandomUnderSampler(random_state=SEED)
    X_res, y_res = rus.fit_resample(X_train, y_train)
   
    # prepare to lgbm fitting
    train_set = lgb.Dataset(data=X_res, label=y_res, 
                            feature_name=X_res.columns.tolist())
    valid_set = lgb.Dataset(data=X_clean_valid, label=y_valid,
                            feature_name=X_res.columns,
                            reference=train_set)
    if not params:
        params = {'num_leaves': 31, 'force_col_wise': True}
    
    # params of the lgbm-model
    lgbm = lgb.train(params=params, 
                      train_set=train_set, 
                      valid_sets=[valid_set],
                      num_boost_round=100, 
                      early_stopping_rounds=25,
                      feval=roc_auc_feval,
                      verbose_eval=-1)
    
    return lgbm

# lgbm = fit(X_clean_train, y_train)
#%% UNDER SAMPLING
rus = RandomUnderSampler(random_state=SEED)
X_res, y_res = rus.fit_resample(X_clean_train, y_train) 
#%% LGBM fitting for extract feature importancies
# prepare data for lgbm fitting 
train_set = lgb.Dataset(data=X_res, label=y_res, 
                        feature_name=X_res.columns.tolist())
valid_set = lgb.Dataset(data=X_clean_valid, label=y_valid,
                        feature_name=X_res.columns,
                        reference=train_set)
# params of the lgbm-model
params = {
    'num_leaves': 31
    }

# fitting the model
lgbm = lgb.train(params=params, 
                  train_set=train_set, 
                  valid_sets=[valid_set],
                  num_boost_round=100, 
                  early_stopping_rounds=25,
                  feval=roc_auc_feval,
                  verbose_eval=10)

preds = lgbm.predict(X_res)
score = roc_auc_score(y_res, preds)
print(score.round(4))

preds = lgbm.predict(X_clean_train)
score = roc_auc_score(y_train, preds)
print(score.round(4))

preds = lgbm.predict(X_clean_valid)
score = roc_auc_score(y_valid, preds)
print(score.round(4))
#%% PLOT FEATURE IMPORTANCE

ax = lgb.plot_importance(lgbm, max_num_features=200, 
                         figsize=(15, 15))
plt.savefig('media/lgbm.plot_importance.png')
 #%% FEATURE IMPORTANCE TO DATAFRAME. 
 
df_fi = pd.DataFrame({'auc': lgbm.feature_importance(), 'feats': X_res.columns})
df_fi.index = X_res.columns
df_fi.sort_values(by='auc', ascending=False, inplace=True)
#%% Sequential forward selection using feature_importance

dct_res, dct_train, dct_valid, dct_test, iters = {}, {}, {}, {}, {}
n = len(df_fi.index)  # number of feature_importance
feats_selected = [] # feats that will be used in forward selection

for feat in tqdm(df_fi.index[:n]):
    
    feats_selected.append(feat)
    length = len(feats_selected)
    
    lgbm = fit(X_res[feats_selected], y_res)
    iters[length] = lgbm.best_iteration
    
    preds = lgbm.predict(X_res[feats_selected])
    score = roc_auc_score(y_res, preds)
    dct_res[length] = score.round(4)
    
    preds = lgbm.predict(X_res[feats_selected])
    score = roc_auc_score(y_res, preds)
    dct_train[length] = score.round(4)
    
    preds = lgbm.predict(X_clean_valid[feats_selected])
    score = roc_auc_score(y_valid, preds)
    dct_valid[length] = score.round(4)

dct = {'res': dct_res, 'train': dct_train, 'valid': dct_valid, 'n_estim': iters}
df_res = pd.DataFrame(dct)

df_res.sort_values(by='valid', ascending=False, inplace=True)
#%% the best result on valid set
best = df_res.head(1).index.values[0]
feats_to_save = df_fi[:best].index 
#%% save my clean train, valid and test sets for tuning model
train_set_clean = {
    'X_clean_train': X_clean_train[feats_to_save],
    'y_train': y_train,
    'feats_to_save': feats_to_save
    }
with open('pkl/data_clean_train.pkl', 'wb') as file:
    pkl.dump(train_set_clean, file)
    
valid_set_clean = {
    'X_clean_valid': X_clean_valid[feats_to_save],
    'y_valid': y_valid
    }
with open('pkl/data_clean_valid.pkl', 'wb') as file:
    pkl.dump(valid_set_clean, file)    

test_set_clean = {
    'X_clean_test': X_clean_test[feats_to_save],
    }
with open('pkl/data_clean_test.pkl', 'wb') as file:
    pkl.dump(test_set_clean, file)