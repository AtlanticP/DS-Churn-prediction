import pickle as pkl 
import pandas as pd
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np 
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope as ho_scope
from functools import partial
from itertools import product
from tqdm import tqdm

SEED = 32
#%% load data

with open('pkl/data_clean_train.pkl', 'rb') as file:
    DATA = pkl.load(file)
    
X_train = DATA['X_clean_train']
y_train = DATA['y_train']    
feats = DATA['feats_to_save']

with open('pkl/data_clean_valid.pkl', 'rb') as file:
    DATA = pkl.load(file)
    
X_valid = DATA['X_clean_valid']
y_valid = DATA['y_valid']    

with open('pkl/data_clean_test.pkl', 'rb') as file:
    DATA = pkl.load(file)
    
X_test = DATA['X_clean_test']
#%% FEVAL FOR LGBM early_stopping_rounds
def roc_auc_feval(preds, train_data):
    y = train_data.get_label()
    score = roc_auc_score(y, preds)
    return 'roc_auc', score, True
#%% OBJECTIVE function for HYPEROPT

def objective(params, X_train, y_train):
    
    kf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    
    scores = []   # score at each iteration
    
    try: 
        for itrain, ivalid in kf.split(X_train, y_train):
            
            rus = RandomUnderSampler(random_state=SEED) 
            X_res, y_res = rus.fit_resample(X_train.iloc[itrain, :], y_train[itrain])
            gbm = lgb.LGBMClassifier(**params, n_estimators=1000)
            gbm.fit(X_res, y_res, 
                    eval_set=[(X_train.iloc[ivalid, :], y_train[ivalid])],
                    early_stopping_rounds=25, 
                    verbose=-1)
            
            score = dict(gbm.best_score_['valid_0'])['auc']
            scores.append(score)
        
        return {'loss': -np.mean(scores), 'params': params, 'status': STATUS_OK}
    
    except ValueError:
        
        return {'loss': None, 'params': params, 'status': STATUS_FAIL}

#%% HYPEROPT optimization

space = {
    # 'objective': 'binary',   ########### delete
    # 'force_col_wise': True,
    'class_weight':     hp.choice('class_weight', [None, 'balanced']),
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.01, 0.2, 0.02)),
    'num_leaves':       ho_scope.int(hp.quniform('num_leaves', 8, 128, 2)),
    # 'max_depth':        hp.choice('max_depth',        np.arange(1, 20, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.5, 1),
    'reg_alpha':        hp.uniform('reg_alpha', 0.01, 1.0),
    'reg_lambda':       hp.uniform('reg_lambda', 0.01, 1.0),
    'metric': 'auc'
}

trials = Trials()
best = fmin(fn=partial(objective, X_train=X_train, y_train=y_train), 
            space=space, algo=tpe.suggest, 
            max_evals=2, trials=trials, 
            show_progressbar=True, 
            rstate=np.random.RandomState(SEED))
res = trials.results
res = [{**i, **i['params']} for i in res]
df_hyp = pd.DataFrame(res)
df_hyp['loss'] = df_hyp['loss'].apply(abs)
df_hyp.sort_values(by='loss', ascending=False, inplace=True)
df_hyp.drop(['params', 'status'], axis=1, inplace=True)
#%%
# with open('pkl/hyperopt_100.pkl', 'wb') as file:
#     pkl.dump(df_hyp, file)    

with open('pkl/hyperopt_100.pkl', 'rb') as file:
    df_hyp = pkl.load(file)    
#%% PREDICTION with best params of HYPEROPT :0.715438 DELETE

best_10 = df_hyp.drop('loss', axis=1).head(10).T.to_dict()

rus = RandomUnderSampler(random_state=SEED) 
X_res, y_res = rus.fit_resample(X_train, y_train)
ress = {}   # auc of train, valid sets

for key, params in best_10.items():

    train_set = lgb.Dataset(data=X_res, label=y_res)
    valid_set = lgb.Dataset(data=X_valid, label=y_valid, 
                            reference=train_set)
    
    lgbm = lgb.train(params=params, train_set=train_set, 
                      valid_sets=valid_set,
                      early_stopping_rounds=25, 
                      num_boost_round=5000,
                      feval=roc_auc_feval,
                      verbose_eval=-1)
#%% train BEST model  probably DELETE
params_hyp_best = df_hyp.drop('loss', axis=1).iloc[0, :].T.to_dict()

rus = RandomUnderSampler(random_state=SEED) 
X_res, y_res = rus.fit_resample(X_train, y_train)
train_set = lgb.Dataset(data=X_res, label=y_res)
valid_set = lgb.Dataset(data=X_valid, label=y_valid, 
                        reference=train_set)

lgbm_best = lgb.train(params_hyp_best, train_set, num_boost_round=5000,
                      valid_sets=valid_set, valid_names=feats,
                      feval=roc_auc_feval, early_stopping_rounds=25,
                      verbose_eval=False)

n = lgbm_best.best_iteration  # number of estimators
lgbm_best2 = lgb.LGBMClassifier( **params_hyp_best)
lgbm_best2.fit(X_res, y_res, eval_set=[(X_valid, y_valid)],
               early_stopping_rounds=25)

preds = lgbm_best.predict(X_res)
score = roc_auc_score(y_res, preds)
print('auc score of best_hyp_model on resampled: ', score.round(4))

preds = lgbm_best2.predict(X_res)
score = roc_auc_score(y_res, preds)
print('auc score of best_hyp_model on resampled 2: ', score.round(4))

preds = lgbm_best.predict(X_train)
score = roc_auc_score(y_train, preds)
print('auc score of best_hyp_model on train: ', score.round(4))

preds = lgbm_best2.predict(X_train)
score = roc_auc_score(y_train, preds)
print('auc score of best_hyp_model on train2: ', score.round(4))

preds = lgbm_best.predict(X_valid)
score = roc_auc_score(y_valid, preds)
print('auc score of best_hyp_model on valid: ', score.round(4))

preds = lgbm_best2.predict(X_valid)
score = roc_auc_score(y_valid, preds)
print('auc score of best_hyp_model on valid2: ', score.round(4))
#%% TO DELETE

params_hyp_best = df_hyp.drop(['loss'], axis=1).iloc[0, :].to_dict()

rus = RandomUnderSampler(random_state=SEED) #%%
X_res, y_res = rus.fit_resample(X_train, y_train)


gbm = lgb.LGBMClassifier(**params_hyp_best, n_estimators=1000)
gbm.fit(X_res, y_res, 
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=25, 
        verbose=-1)

preds = gbm.booster_.predict(X_test, raw_score=False)
print(preds[:3])

df_pred_test = pd.DataFrame({'Id': np.arange(len(preds)), 'result': preds})
df_pred_test.to_csv('y_test.csv', sep=',', index=False)
