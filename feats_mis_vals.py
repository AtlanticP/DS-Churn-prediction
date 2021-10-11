# import os
import numpy as np
import pandas as pd
# import missingno as msno
import pickle as pkl
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
#%% load data

with open('pkl/df_out_train.pkl', 'rb') as file:
    DATA = pkl.load(file)
df_train = DATA['df_train']
feats_num = DATA['feats_num']
feats_cat = DATA['feats_cat']

with open('pkl/df_out_valid.pkl', 'rb') as file:
    DATA = pkl.load(file)
df_valid = DATA['df_valid']

with open('pkl/X_out_test.pkl', 'rb') as file:
    DATA = pkl.load(file)
X_test = DATA['X_test']

print('shape of df_train:', df_train.shape)
print('shape of df_valid:', df_valid.shape)
print('shape of X_test:', X_test.shape)

trg = 'labels'

X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_valid = df_valid.iloc[:, :-1]
y_valid = df_valid.iloc[:, -1]
#%% distribution of target
df_train.iloc[:, -1].value_counts().to_dict() 
#{-1.0: 12690, 1.0: 1033}

(df_train.iloc[:, -1].value_counts()/len(df_train)).round(3).to_dict()
# {-1.0: 0.925, 1.0: 0.075}
#%% Handling missing values
print('Handling missing values')
#%% Visualize Missing values
# fig = msno.matrix(df)
# fig = fig.get_figure()
# fig.savefig('media/msno.png') 
#%% HANDLING NUMERICAL MIS. DATA. 
#%% Relationship between MVs and target

# plot_relations(df, feats_num, 'rels_num')
#%% KNN Imputation, fit and transform train_data
print('\nNumerical data. KNN imputation')

imputer = KNNImputer()
imputer.fit(X_train[feats_num])

X_train[feats_num] = imputer.transform(X_train[feats_num])
X_valid[feats_num] = imputer.transform(X_valid[feats_num])
X_test[feats_num] = imputer.transform(X_test[feats_num])

mes = X_train[feats_num].isna().any().any()
print('X_train numerical feats isna any: ', mes)
mes = X_train[feats_num].isna().any().any()
print('X_valid numerical feats isna any:', mes)
mes = X_test[feats_num].isna().any().any()
print('X_test numerical feats isna any:', mes)

#%% HANDLING MISSING VALUES. CATEGORICAL DATA. 
#%% Relationship between MVs and target

# plot_relations(df, feats_cat, file_name='MV_targ.png')
#%%
feats_to_del = [191, 201, 213, 215, 224]   # features to delete
feats_to_del = [f'Var{col}' for col in feats_to_del]

feats_cat = list(filter(lambda feat: feat not in feats_to_del, feats_cat))

print('shape of X_train:', X_train.shape)
print('shape of X_valid:', X_valid.shape)
print('shape of X_test:', X_test.shape)
#%% Relationship between MVs and target. Partially deleted feats.

# plot_relations(df, feats_cat, file_name='MV_targ2.png')
#%% MCNAR: using these feats create new feat: "isna_Var194" and in original feat replace feat with 'rare' - category.
print('\nCategorical data. Creating new features. And replacing nan values with "rare" category')
feats_to_new = [194, 197, 199, 200, 206, 214, 223, 225, 229]
feats_to_new = [f'Var{col}' for col in feats_to_new]

for feat in feats_to_new:
    
    new = 'isna_' + feat
    feats_cat.append(new)
    
    X_train[new] = np.where(X_train[feat].isna(), 'missing', 'existing')
    X_train[feat] = np.where(X_train[feat].isna(), 'rare', X_train[feat])
    
    X_valid[new] = np.where(X_valid[feat].isna(), 'missing', 'existing')
    X_valid[feat] = np.where(X_valid[feat].isna(), 'rare', X_valid[feat])
    
    X_test[new] = np.where(X_test[feat].isna(), 'missing', 'existing')
    X_test[feat] = np.where(X_test[feat].isna(), 'rare', X_test[feat])

print('\nCategorical data. Creating new features replacing nan values with "rare" category')
print('shape of df_train:', X_train.shape)
print('shape of df_valid:', X_valid.shape)
print('shape of df_test:', X_test.shape)
#%% MCAR: inpute 'missing'
mask = X_train[feats_cat].isna().any()
feats_cat_na = X_train[feats_cat].columns[mask]    
#%%  Relationship between MVs and target. Partially deleted feats.
# plot_relations(df, feats_cat_na, ncols=4, file_name='MV_targ3.png') 
#%% MCAR: most frequent strategy

for feat in feats_cat:

    top = X_train[feat].value_counts().nlargest(1).index.values[0]
    X_train[feat].fillna(top, inplace=True)
    X_valid[feat].fillna(top, inplace=True)
    X_test[feat].fillna(top, inplace=True)

print('\nCategorical data. MCAR: most_frequent')
print('shape of X_train:', X_train.shape)
print('shape of X_valid:', X_valid.shape)
print('shape of X_test:', X_test.shape)      
#%% check nan
isna = X_train[feats_num + feats_cat].isna().any().any() # False
print('train is NaN any:', isna)
isna = X_valid[feats_num + feats_cat].isna().any().any() # False
print('valid is NaN any:', isna)
isna = X_test[feats_num+feats_cat].isna().any().any() # False
print('test is NaN any:', isna)
#%% convert target type to int
y_train = y_train.astype(int)
y_valid = y_valid.astype(int)
#%% save my train, valid and test sets for further processing

train_set_filled = {
    'df_train': pd.concat([X_train[feats_num+feats_cat], y_train], axis=1),
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }
with open('pkl/df_filled_train.pkl', 'wb') as file:
    pkl.dump(train_set_filled, file)
   
valid_set_filled = {
    'df_valid': pd.concat([X_valid[feats_num+feats_cat], y_valid], axis=1),
    }
with open('pkl/df_filled_valid.pkl', 'wb') as file:
    pkl.dump(valid_set_filled, file)    

test_set_filled = {
    'X_test': X_test[feats_num+feats_cat],
    }
with open('pkl/X_filled_test.pkl', 'wb') as file:
    pkl.dump(test_set_filled, file)  









