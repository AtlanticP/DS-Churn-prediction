import pickle as pkl 
import pandas as pd
import math
import os

# import missingno as msno
import pickle as pkl
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.impute import KNNImputer
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from outliers import OUT_handler  # my own class to handle outliers
plt.style.use('seaborn')

SEED = 32
#%% Relationship between MVs and target
def plot_relations(df, feats, file_name, ncols=10):
    
    dft = df.copy()
    n = len(feats)
    
    fig, axes = plt.subplots(nrows=np.ceil(n/ncols).astype(int), 
                              ncols=ncols, figsize=(3*ncols, np.ceil(n/ncols)*3))
    
    for idx, feat in enumerate(feats):
        
        dft[feat] = np.where(dft[feat].isna(), 1, 0)
        groups = dft.groupby(trg)
        ax=axes[idx//ncols, idx%ncols]
        ax.set_title(feat)
        
        for name, group in groups:
            ax.bar(name, group[feat].mean(), label=name, align='center')
      
    fig.suptitle('Relationship between MVs and target')
    fig.tight_layout()
    fig.savefig(f'media/{file_name}')

#%%
PATH_DIR = '../../../bases/telecom-clients-prediction2'
FILE_NAME_TRAIN = 'orange_small_churn_train_data.csv'
FILE_NAME_TEST = 'orange_small_churn_test_data.csv'
PATH_FILE_TRAIN = os.path.join(PATH_DIR, FILE_NAME_TRAIN)
PATH_FILE_TEST = os.path.join(PATH_DIR, FILE_NAME_TEST)
#%%
df_train = pd.read_csv(PATH_FILE_TRAIN, index_col='ID')

feats_num = df_train.columns[:190]
feats_cat = df_train.columns[190:-1]
trg = df_train.columns[-1]
print('shape of raw train data:', df_train.shape)     # (18299, 231)

X_test = pd.read_csv(PATH_FILE_TEST, index_col='ID')
print('shape of raw test data:', X_test.shape)     # (18299, 231)
#%% drop elements where target is nan
df_train.drop(np.where(df_train.iloc[:, -1].isna())[0], axis=0, inplace=True)
print('train: shape of nan-target: ', df_train.shape)  # (18298, 231)
#%% split data and save test set
print('split data to train, valid')
X_train, X_valid, y_train, y_valid = train_test_split(
                    df_train.iloc[:, :-1], 
                    df_train.iloc[:, -1],                    
                    test_size=0.25, shuffle=True, 
                    stratify=df_train.iloc[:, -1], random_state=SEED)

df_train = pd.concat([X_train, y_train], axis=1)
print('shape of X_train:', X_train.shape)
print('shape of X_valid:', X_valid.shape)
print('shape of X_test:', X_test.shape, '\n')
#%% Remove completly nan columns 
feats_num = X_train[feats_num].dropna(how='all', axis=1).columns.tolist()
feats_cat = X_train[feats_cat].dropna(how='all', axis=1).columns.tolist()
feats = feats_num + feats_cat 
print('shape of X_train:', X_train[feats].shape)
#%% Drop feats with nan more than 0.925
print('Length of numerical feats before dropping', len(feats_num))

prop = (df_train.iloc[:, -1].value_counts()/len(df_train)).round(3).to_dict()
# {-1.0: 0.925, 1.0: 0.075}
print('Proportions of churn', prop)

mask = (X_train[feats_num].isna().mean() < 0.925)
feats_num = X_train[feats_num].columns[mask].tolist()
print('Length of numeric vals after dropping: ', len(feats_num))

feats = feats_cat + feats_num
#%% feats with high cardinality to categorical feats

txt = 'feats with high cardinality to categorical feats:'
print(txt)

feats_tocat = []
for feat in feats_num:
    card = df_train[feat].value_counts().size   # cardinality
    if card < 30:
        feats_tocat.append(feat)
        print(feat, card)
        
feats_num = list(filter(lambda feat: feat not in feats_tocat, feats_num))        
feats_cat = feats_cat + feats_tocat
#%% NUMERICAL FEATURES
#%% plot kde of train, valid, test sets

n = len(feats_num)
ncols = 5
nrows = int(np.ceil(n/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))

for idx, feat in enumerate(feats_num[:n]):
    
    ax = axes[idx//ncols, idx%ncols]
    sns.kdeplot(x=X_train[feat], ax=ax, alpha=0.5, label='X_train')
    sns.kdeplot(x=X_valid[feat], ax=ax, alpha=0.5, label='X_valid')
    sns.kdeplot(x=X_test[feat], ax=ax, alpha=0.5, label='X_test')
    ax.set_title(feat)
    ax.legend()

fig.tight_layout()    
fig.savefig('media/feats_num_kde_tr_vl_ts.png')
#%% plot boxplot, kde, boxplot log(x), kde log(x)  

nrows = len(feats_num)
ncols = 4

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
feat = feats_num[0]

for idx, feat in enumerate(feats_num[:nrows]):
    
    sns.boxplot(y=X_train[feat], ax=axes[idx, 0], x=y_train)
    axes[idx, 0].set_title(feat)
    sns.kdeplot(x=feat, data=X_train, hue=y_train, ax=axes[idx, 1])
    axes[idx, 1].set_title(feat)
    sns.boxplot(y=X_train[feat].apply(np.log), ax=axes[idx, 2], x=y_train)
    axes[idx, 2].set_title(f'ln{feat}')
    sns.kdeplot(x=X_train[feat].apply(lambda x: np.log(x+1e-5)), hue=y_train, ax=axes[idx, 3])
    axes[idx, 3].set_title(f'log({feat})')

fig.tight_layout()
fig.savefig('media/log_boxplot_kde.png')
#%% feats with log-transform
feats_tolog = [6, 21, 22, 25, 94, 109, 119, 123, 125, 160]
feats_tolog = [f'Var{var}' for var in feats_tolog]

feats_tolater = [13, 24, 74, 83, 85, 112, 140, 149]
feats_tolater = [f'Var{var}' for var in feats_tolater]

feats_torem = set(feats_num)-set(feats_tolog)-set(feats_tolater)
feats_torem = sorted(feats_torem, key=lambda x: int(x[3:]))
print('feats to log transform:', feats_torem, '\n')

for feat in feats_tolog:
    X_train[feat] = X_train[feat].apply(np.log)
#%% HANDLING OUTLIERS (IQR method)

nrows = len(feats_num)
# nrows = 10
ncols = 4

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))

for idx, feat in enumerate(feats_num[:nrows]):
    
    sample = X_train.loc[:, feat] 
    ax=axes[idx, 0]
    sns.boxplot(y=sample , ax=axes[idx, 0])
    axes[idx, 0].set_title(f'{feat}. With outliers')
    sns.kdeplot(x=sample, ax=axes[idx, 1])
    axes[idx, 1].set_title(f'{feat}. With outliers')

    # transformation
    handler = OUT_handler(method='iqr')
    handler.fit(sample.values)
    sample = handler.transform(sample) 

    sns.boxplot(y=sample, ax=axes[idx, 2])
    axes[idx, 2].set_title(f'{feat}. Without outliers')
    sns.kdeplot(x=sample, ax=axes[idx, 3])
    axes[idx, 3].set_title(f'{feat}. Without outliers')
    
fig.tight_layout()
fig.savefig('media/outliers_handle.png')    