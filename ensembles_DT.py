
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, recall_score 
from sklearn.metrics import accuracy_score, precision_score

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

from category_encoders import BinaryEncoder
from tqdm import tqdm
import pickle as pkl
import numpy as np 
import pandas as pd
 
SEED = 32
#%%
with open('pkl/df_train_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df = data['df_train']
feats_num = data['feats_num']
feats_cat = data['feats_cat']
#%%
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)

ifeats_num = slice(0, 41)
ifeats_cat = slice(41, X.shape[1])
#%%
transformer = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), ifeats_num),
            ('cat', BinaryEncoder(), ifeats_cat)],
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
        pipe.fit(X[itrain], y[itrain])
        preds = pipe.predict(X[itest])
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
print(results)
