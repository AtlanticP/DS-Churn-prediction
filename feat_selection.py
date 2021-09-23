import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from ohe_topN import OHE_topN  
import pickle as pkl  
import pandas as pd
from tqdm import tqdm
import shap 
import matplotlib.pyplot as plt

import numpy as np 
import math

SEED = 32
#%% load filled-nnan data: train, valid, test set
with open('pkl/df_train_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df_train = data['df_train']
feats_num = data['feats_num']
feats_cat = data['feats_cat']
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]

# load valid set
with open('pkl/df_valid_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df_valid = data['df_valid']
X_valid, y_valid = df_valid.iloc[:, :-1], df_valid.iloc[:, -1]

# load test set
with open('pkl/df_test_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df_test = data['df_test']
X_test = df_test.copy()
#%% FEATURE TRANSFORMATION
print('train transformation')
# numerical features
X_num_train = StandardScaler().fit_transform(X_train[feats_num])
X_num_train = pd.DataFrame(X_num_train, columns=feats_num)
X_num_valid = StandardScaler().fit_transform(X_valid[feats_num])
X_num_valid = pd.DataFrame(X_num_valid, columns=feats_num)
X_num_test = StandardScaler().fit_transform(X_test[feats_num])
X_num_test = pd.DataFrame(X_num_test, columns=feats_num)

# categorical features
ohe = OHE_topN(top_n=10)
ohe.fit(X_train[feats_cat])
X_cat_train = ohe.fit_transform(X_train[feats_cat])
X_cat_valid = ohe.transform(X_valid[feats_cat])
X_cat_test = ohe.transform(X_test[feats_cat])
feats_cat = X_cat_train.columns

# concatenatenate numerical and categorical features
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

lgbm = fit(X_clean_train, y_train)
#%% UNDER SAMPLING
rus = RandomUnderSampler(random_state=SEED)
X_res, y_res = rus.fit_resample(X_clean_train, y_train) 
#%% prepare to lgbm fitting !!!!!!!!!!!!!!!!!
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
#%% FEATURE IMPORTANCE TO DATAFRAME. EDA.
lgbm = fit(X_res, y_res)

df_fi = pd.Series(lgbm.feature_importance())
df_fi.index = X_res.columns
df_fi.sort_values(ascending=False, inplace=True)

n = X_clean_train.shape[1]
ncols = 4
nrows = math.ceil(n/ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))

for idx, feat in enumerate(df_fi.index[:n]):
    
    ax = axes[idx//4, idx%4]
    
    X_clean_train.loc[:, feat].hist(ax=ax, color='b', alpha=0.4, label='train')
    X_clean_valid.loc[:, feat].hist(ax=ax, color='r', alpha=0.5, label='valid')
    X_clean_test.loc[:, feat].hist(ax=ax, color='g', alpha=0.4, label='test')
    ax.legend()
    ax.set_title(feat)

fig.tight_layout()
fig.savefig('media/distr.hist.train_valid_test.png')
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

#%% save my clean train, valid and test sets for tuning model

# the best result on valid set
best = df_res.head(1).index.values[0]
feats_to_save = df_fi[:best].index 

train_set_clean = {
    'X_clean_train': X_clean_train[feats_to_save],
    'y_train': y_train,
    'feats_to_save': feats_to_save
    }
with open('pkl/df_train_clean.pkl', 'wb') as file:
    pkl.dump(train_set_clean, file)
    
valid_set_clean = {
    'X_valid_clean': X_clean_valid[feats_to_save],
    'y_valid': y_valid
    }
with open('pkl/df_valid_clean.pkl', 'wb') as file:
    pkl.dump(valid_set_clean, file)    

test_set_clean = {
    'X_test_clean': X_clean_test[feats_to_save],
    }
with open('pkl/df_test_clean.pkl', 'wb') as file:
    pkl.dump(test_set_clean, file)
#%%
X_res[feats_to_save]