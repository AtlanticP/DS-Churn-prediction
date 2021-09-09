import numpy as np 
from functools import partial
import lightgbm as lgb
from hyperopt import hp, fmin, Trials, tpe, STATUS_FAIL, STATUS_OK
from imblearn.under_sampling import RandomUnderSampler
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
X_train, y_train = df.iloc[:, :-1], df.iloc[:, -1]
#%%
with open('pkl/df_test_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df_test = data['df_test']
feats_num = data['feats_num']
feats_cat = data['feats_cat']
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
#%% FEATURE TRANSFORMATION

transformer = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), feats_num),
        ('cat', OHE_topN(top_n=10), feats_cat)],
    remainder = 'passthrough',
    n_jobs = -1,
    verbose = True
    )
print('train transformation')
X_train_clean = transformer.fit_transform(X_train)

print('test transformation')
X_test_clean = transformer.transform(X_test)
#%% load parameters of best model
with open('pkl/hp_res.pkl', 'rb') as file:    
    df_hp = pkl.load(file)
#%%
print('fitting model')

res = {}

n = 10  # evaluate top n models

for i in df_hp.index[:n]:
    
    print('index:', i)
    
    params_all = df_hp.drop(['loss'], axis=1).T.to_dict()
    params = params_all[i]
    
    lgbm = lgb.LGBMClassifier(**params)
    rus = RandomUnderSampler(random_state=SEED)
           
    X_res, y_res = rus.fit_resample(X_train_clean, y_train)
    lgbm.fit(X_res, y_res, verbose=-1)
        
    preds = lgbm.predict(X_test_clean)
    roc = roc_auc_score(y_test, preds)
    print('Test roc_auc:', roc.round(3), '\n') 
    
    res[i] = {'test_roc_auc': roc, 'train_roc_auc': df_hp['loss'].to_dict()[i]}

df_res = pd.DataFrame(res).T
df_res.sort_values(by=['test_roc_auc', 'train_roc_auc'], ascending=[False, False], inplace=True)
print(df_res)
#%% save results
with open('pkl/df_res.pkl', 'wb') as file:
    pkl.dump(df_res, file)
    
with open('pkl/df_res.pkl', 'rb') as file:
    df_res = pkl.load(file) 

    

 
    
    
    
    
    
    
    
    
    