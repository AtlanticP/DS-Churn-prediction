import os
import numpy as np
import pandas as pd
# import missingno as msno
import pickle as pkl
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt
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
feats = df_train.columns
feats_num = df_train.columns[:190]
feats_cat = df_train.columns[190:-1]
trg = df_train.columns[-1]
print('shape of raw train data:', df_train.shape)     # (18299, 231)

df_test = pd.read_csv(PATH_FILE_TEST, index_col='ID')
print('shape of raw train data:', df_test.shape)     # (18299, 231)
#%% drop elements where target is nan
df_train.drop(np.where(df_train.iloc[:, -1].isna())[0], axis=0, inplace=True)
print('train: shape of nan-target: ', df_train.shape)  # (18298, 231)
#%%
X_train = df_train.iloc[:, :-1].values  
y_train = df_train.iloc[:, -1].values
#%% split data and save test set
X_train, X_valid, y_train, y_valid = train_test_split(
                    X_train, y_train,                    
                    test_size=0.25, shuffle=True, 
                    stratify=y_train, random_state=SEED)

df_train = pd.DataFrame(np.c_[X_train, y_train], columns=feats)
df_valid = pd.DataFrame(np.c_[X_valid, y_valid], columns=feats)

with open('pkl/df_valid.pkl', 'wb') as file:
    pkl.dump(df_test, file)

print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of df_test:', df_test.shape)
#%%
df_train.iloc[:, -1].value_counts().to_dict() 
#{-1.0: 12690, 1.0: 1033}

(df_train.iloc[:, -1].value_counts()/len(df_train)).round(3).to_dict()
# {-1.0: 0.925, 1.0: 0.075}
#%%
print('Handling missing values')
#%% Visualize Missing values
# fig = msno.matrix(df)
# fig = fig.get_figure()
# fig.savefig('media/msno.png') 
#%% Remove completly nan columns 
feats_num = df_train[feats_num].dropna(how='all', axis=1).columns
feats_cat = df_train[feats_cat].dropna(how='all', axis=1).columns
#%% HANDLING NUMERICAL MIS. DATA. 
#%%Drop feats with nan more than 0.925
mask = (df_train[feats_num].isna().mean() >= 0.925)
feats_to_del = df_train[feats_num].loc[:, mask].columns
feats_num = list(filter(lambda x: x not in feats_to_del, feats_num))
df_train.drop(feats_to_del, inplace=True, axis=1)
df_valid.drop(feats_to_del, inplace=True, axis=1)
df_test.drop(feats_to_del, inplace=True, axis=1)
#%% Relationship between MVs and target

# plot_relations(df, feats_num, 'rels_num')
#%% KNN Imputation, fit and transform train_data

# imputer = KNNImputer()
# imputer.fit(df_train[feats_num])

# save imputer
# with open('pkl/imputer_knn.pkl', 'wb') as file:
#     pkl.dump(imputer, file)

with open('pkl/imputer_knn.pkl', 'rb') as file:
    imputer = pkl.load(file)

df_train[feats_num] = imputer.transform(df_train[feats_num])
df_valid[feats_num] = imputer.transform(df_valid[feats_num])
df_test[feats_num] = imputer.transform(df_test[feats_num])

print('\nNumerical data. KNN imputation')
print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of df_test:', df_test.shape)
#%% HANDLING MISSING VALUES. CATEGORICAL DATA. 
#%% Relationship between MVs and target

# plot_relations(df, feats_cat, file_name='MV_targ.png')
#%%
feats_to_del = [191, 201, 213, 215, 224]   # features to delete
feats_to_del = [f'Var{col}' for col in feats_to_del]

feats_cat = list(filter(lambda x: x not in feats_to_del, feats_cat))
df_train.drop(feats_to_del, axis=1, inplace=True)
df_valid.drop(feats_to_del, axis=1, inplace=True)
df_test.drop(feats_to_del, axis=1, inplace=True)

print('\nCategorical data. Creating new features replacing nan values with "rare" category')
print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of df_test:', df_test.shape)
#%% Relationship between MVs and target. Partially deleted feats.

# plot_relations(df, feats_cat, file_name='MV_targ2.png')
#%% MCNAR: using these feats create new feat: "isna_Var194" and in original feat replace feat with 'rare' - category.

feats_to_new = [194, 197, 199, 200, 206, 214, 223, 225, 229]
feats_to_new = [f'Var{col}' for col in feats_to_new]

for feat in feats_to_new:
    
    new = 'isna_' + feat
    feats_cat.append(new)
    
    df_train[new] = np.where(df_train[feat].isna(), 'missing', 'existing')
    df_train[feat] = np.where(df_train[feat].isna(), 'rare', df_train[feat])
    
    df_valid[new] = np.where(df_valid[feat].isna(), 'missing', 'existing')
    df_valid[feat] = np.where(df_valid[feat].isna(), 'rare', df_valid[feat])
    
    df_test[new] = np.where(df_test[feat].isna(), 'missing', 'existing')
    df_test[feat] = np.where(df_test[feat].isna(), 'rare', df_test[feat])

print('\nCategorical data. Creating new features replacing nan values with "rare" category')
print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of df_test:', df_test.shape)
#%% MCAR: inpute 'missing'
mask = df_train[feats_cat].isna().any()
feats_cat_na = df_train[feats_cat].columns[mask]    
#%%  Relationship between MVs and target. Partially deleted feats.
# plot_relations(df, feats_cat_na, ncols=4, file_name='MV_targ3.png') 
#%% MCAR: most frequent strategy

for feat in feats_cat:

    top = df_train[feat].value_counts().nlargest(1).index.values[0]
    df_train[feat].fillna(top, inplace=True)
    df_valid[feat].fillna(top, inplace=True)
    df_test[feat].fillna(top, inplace=True)

print('\nCategorical data. MCAR: most_frequent')
print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of df_test:', df_test.shape)      
#%% check nan
isna = df_train[feats_num + feats_cat + ['labels']].isna().any().any() # False
print('train is NaN:', isna)
isna = df_valid[feats_num + feats_cat + ['labels']].isna().any().any() # False
print('valid is NaN:', isna)
isna = df_test[feats_num+feats_cat].isna().any().any() # False
print('test is NaN:', isna)
#%% convert target type to int
df_train[trg] = df_train[trg].astype(int)
df_valid[trg] = df_valid[trg].astype(int)
#%% save my train, valid and test sets for further processing
train_set_filled = {
    'df_train': df_train[feats_num+feats_cat+['labels']],
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }
with open('pkl/df_train_filled.pkl', 'wb') as file:
    pkl.dump(train_set_filled, file)
    
valid_set_filled = {
    'df_valid': df_valid[feats_num+feats_cat+['labels']],
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }
with open('pkl/df_valid_filled.pkl', 'wb') as file:
    pkl.dump(valid_set_filled, file)    

test_set_filled = {
    'df_test': df_test[feats_num+feats_cat],
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }
with open('pkl/df_test_filled.pkl', 'wb') as file:
    pkl.dump(test_set_filled, file)
#%%
import math 
import matplotlib.pyplot as plt 

n = 10
ncols = 4
nrows = math.ceil(n/ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*3, ncols*3))

for idx, feat in enumerate(df_train.columns[:n]):
    
    ax = axes[idx//4, idx%4]
    
    df_train.loc[:, feat].hist(ax=ax, color='b', alpha=0.4, label='train')
    df_valid.loc[:, feat].hist(ax=ax, color='r', alpha=0.5, label='valid')
    df_test.loc[:, feat].hist(ax=ax, color='g', alpha=0.4, label='test')
    ax.legend()
    ax.set_title(feat)

# fig.legend()
fig.tight_layout()
fig.savefig('media/distr.hist.train_valid_test.png')    
    