import numpy as np 
from functools import partial
import lightgbm as lgb
from hyperopt import hp, fmin, Trials, tpe, STATUS_FAIL, STATUS_OK
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ohe_topN import OHE_topN  
import pickle as pkl  
import pandas as pd
SEED = 32
#%%
with open('pkl/df_train_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df = data['df_train']
feats_num = data['feats_num']
feats_cat = data['feats_cat']
X = df.iloc[:, :-1]
y = df.iloc[:, -1].values  
#%% FEATURE TRANSFORMATION

transformer = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), feats_num),
        ('cat', OHE_topN(top_n=10), feats_cat)],
    remainder = 'passthrough',
    n_jobs = -1,
    verbose = True
    )
X_clean = transformer.fit_transform(X)
     
#%%

def objective(params, X, y):
    
    params['num_leaves'] = int(params['num_leaves'])
    lgbm = lgb.LGBMClassifier(**params)
    rus = RandomUnderSampler(random_state=SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rocs = []    # roc-auc score of the current model

    try:    
        
        for idx, (itrain, itest) in enumerate(skf.split(X, y), start=1):
            
            X_res, y_res = rus.fit_resample(X[itrain, :], y[itrain])

            lgbm.fit(X_res, y_res, early_stopping_rounds=30,
                     eval_set=[(X[itest, :], y[itest])], 
                     eval_metric=['auc'], verbose=-1)
            preds = lgbm.predict(X[itest, :])

            roc = roc_auc_score(y[itest], preds)
            rocs.append(roc)
            
        return {'loss': np.mean(rocs), 
                'params': params, 
                'status': STATUS_OK,
                'n_estimators': lgbm.best_iteration_                
                }    
    
    except ValueError as e:

        return {'loss': None, 'params': params, 'status': STATUS_FAIL}
#%% HYPER OPTIMIZATION

space = {
    'n_estimators': 5000,
    'first_metric_only': True,
    'class_weight':     hp.choice('class_weight', [None, 'balanced']),
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    # 'max_depth':        hp.choice('max_depth',        np.arange(1, 20, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.5, 1),
    'reg_alpha':        hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda':       hp.uniform('reg_lambda', 0.0, 1.0),
    'num_leaves':       hp.quniform('num_leaves', 8, 128, 2)
}
trials = Trials()

best = fmin(fn=partial(objective, X=X_clean, y=y), 
            space=space, algo=tpe.suggest, 
            max_evals=5, 
            trials=trials, 
            rstate=np.random.RandomState(SEED), 
            verbose=-1, 
            show_progressbar=True)
#%%
res = trials.results
res = [{**x, **x['params']} for x in res]

df_res = pd.DataFrame(res)
df_res.drop(labels=['status', 'params', 'first_metric_only'], axis=1, inplace=True)
df_res.sort_values(by='loss', ascending=False, inplace=True)

with open('pkl/hp_res.pkl', 'wb') as file:    
    pkl.dump(df_res, file)