import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from ohe_topN import OHE_topN  
import pickle as pkl  
import pandas as pd
from tqdm import tqdm
import shap 
import matplotlib.pyplot as plt

SEED = 32
#%%
with open('pkl/df_train_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df = data['df_train']
feats_num = data['feats_num']
feats_cat = data['feats_cat']
X_train, y_train = df.iloc[:, :-1], df.iloc[:, -1]
#%% FEATURE TRANSFORMATION
print('train transformation')

X_num = StandardScaler().fit_transform(X_train[feats_num])
X_num = pd.DataFrame(X_num, columns=feats_num)
X_cat = OHE_topN(top_n=10).fit_transform(X_train[feats_cat])
feats_cat = X_cat.columns
X_train_clean = pd.concat([X_num, X_cat], axis=1)

#%% statistic of best models 
with open('pkl/df_hp.pkl', 'rb') as file:
    df_hp = pkl.load(file) 
    
with open('pkl/df_res.pkl', 'rb') as file:
    df_res = pkl.load(file)    
#%%
# best model
idx = df_res.index[0]  
mask = (df_hp.index == idx)
params = df_hp.drop('loss', axis=1).iloc[mask, :].T.squeeze().to_dict()
#%%
rus = RandomUnderSampler(random_state=SEED)
X_res, y_res = rus.fit_resample(X_train_clean, y_train)

#%% SHAP
models = {
    'lgb_params': lgb.LGBMClassifier(**params, random_state=SEED),
    'lgb': lgb.LGBMClassifier(random_state=SEED),
    }
for title, model in models.items():
    
    print('fitting', title)
    model.fit(X_res, y_res)    
    explainer = shap.TreeExplainer(model).shap_values(X_res)
    fig = shap.summary_plot(explainer, X_res, feature_names=X_res.columns,
                      max_display=X_res.shape[1], show=False)                                                  

    plt.savefig(f'media/shap_values_{title}')    
#%%
model = models['lgb_params']
imp = pd.Series(model.feature_importances_)
imp.index = X_res.columns
imp.sort_values(ascending=False, inplace=True)
feats_imp = imp.index
#%% save to testing.py
with open('pkl/feats_imp.pkl', 'wb') as file:
    pkl.dump(feats_imp, file)    
    
    
    
    
    
    