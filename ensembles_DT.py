import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer

from sklearn.metrics import roc_auc_score, recall_score 
from sklearn.metrics import accuracy_score, precision_score

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import pickle as pkl
import numpy as np 
import pandas as pd

SEED = 32
#%%
def get_max_value(df, feats):
    '''search the high number of cardinality of cat feats'''
    
    max_value = 0    
    
    for feat in feats:
    
        size = df[feat].value_counts().size # quantity of categories 
        
        if max_value < size:        
            max_value = size
    
    return max_value
#%%
with open('pkl/df_train_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df = data['df_train']
feats_num = data['feats_num']
feats_cat = data['feats_cat']
#%%
X = df.iloc[:, :-1]
y = df.iloc[:, -1].values.astype(int)
#%%
max_value = get_max_value(df, feats_cat)

transformer = ColumnTransformer(
        transformers = [
            (
                'cat', 
                OrdinalEncoder(handle_unknown='use_encoded_value', 
                               unknown_value=max_value), 
                feats_cat),
            ('num', StandardScaler(), feats_num)
            ],
        remainder = 'passthrough',
        n_jobs = -1,
        verbose = True
        )
#%% Resampling data to balance set
rus = RandomUnderSampler(random_state=SEED)
#%% Models
models = {
    'rf': RandomForestClassifier(random_state=SEED),
    'lgbm': lgb.LGBMClassifier(random_state=SEED),
    }
#%%
results = {}  # overall results 

for title, model in tqdm(models.items()):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    rocs = []    # RocAuc score of the current model
    recs = []    # recall score of the current model
    accs = []
    precs = []
    
    for itrain, itest in skf.split(X, y):
        
        pipe = make_pipeline(transformer, rus, model)
        pipe.fit(X.iloc[itrain, :], y[itrain])
        preds = pipe.predict(X.iloc[itest, :])
        roc = roc_auc_score(y[itest], preds).round(4)
        rec = recall_score(y[itest], preds, pos_label=1).round(4)
        prec = precision_score(y[itest], preds).round(4)
        acc = accuracy_score(y[itest], preds).round(4)
        
        rocs.append(roc)
        recs.append(rec)
        precs.append(prec)
        accs.append(acc)
        
    results[title] = {
        'roc_auc': np.mean(rocs), 
        'recall': np.mean(recs),
        'prec': np.mean(precs),
        'acc': np.mean(accs)        
        }

results = pd.DataFrame(results).T
#%%
print(results.round(3))