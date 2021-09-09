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
FILE_NAME = 'orange_small_churn_train_data.csv'
PATH_FILE = os.path.join(PATH_DIR, FILE_NAME)
#%%
df = pd.read_csv(PATH_FILE, index_col='ID')
feats = df.columns
feats_num = df.columns[:190]
feats_cat = df.columns[190:-1]
trg = df.columns[-1]
print('shape of raw data:', df.shape)     # (18299, 231)
#%% drop elements where target is nan
df.drop(np.where(df.iloc[:, -1].isna())[0], axis=0, inplace=True)
print('shape of nan-target: ', df.shape)  # (18298, 231)
#%%
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values
#%% split data and save test set
X_train, X_test, y_train, y_test = train_test_split(X, y,                    
                    test_size=0.25, shuffle=True, 
                    stratify=y, random_state=SEED)

df = pd.DataFrame(np.c_[X_train, y_train], columns=feats)
df_test = pd.DataFrame(np.c_[X_test, y_test], columns=feats)

with open('pkl/df_test.pkl', 'wb') as file:
    pkl.dump(df_test, file)

print('shape of df_train:', df.shape)
print('shape of df_test:', df_test.shape)
#%%
df.iloc[:, -1].value_counts().to_dict() 
#{-1.0: 12690, 1.0: 1033}

(df.iloc[:, -1].value_counts()/len(df)).round(3).to_dict()
# {-1.0: 0.925, 1.0: 0.075}
#%%
print('Handling missing values')
#%% Visualize Missing values
# fig = msno.matrix(df)
# fig = fig.get_figure()
# fig.savefig('media/msno.png') 
#%% Remove completly nan columns 
feats_num = df[feats_num].dropna(how='all', axis=1).columns
feats_cat = df[feats_cat].dropna(how='all', axis=1).columns
#%% HANDLING NUMERICAL MIS. DATA. 
#%%Drop feats with nan more than 0.925
mask = (df[feats_num].isna().mean() >= 0.925)
feats_to_del = df[feats_num].loc[:, mask].columns
feats_num = list(filter(lambda x: x not in feats_to_del, feats_num))
df.drop(feats_to_del, inplace=True, axis=1)
df_test.drop(feats_to_del, inplace=True, axis=1)
#%% Relationship between MVs and target

# plot_relations(df, feats_num, 'rels_num')
#%% KNN Imputation, fit and transform train_data

# imputer = KNNImputer()
# imputer.fit(df[feats_num])
with open('pkl/imputer_knn.pkl', 'rb') as file:
    imputer = pkl.load(file)

df[feats_num] = imputer.transform(df[feats_num])
df_test[feats_num] = imputer.transform(df_test[feats_num])

print('\nNumerical data. KNN imputation')
print('shape of df_train:', df.shape)
print('shape of df_test:', df_test.shape)

# with open('pkl/knn_imputer.pkl', 'wb') as file:
#     pkl.dump(imputer, file)
#%% HANDLING MISSING VALUES. CATEGORICAL DATA. 
#%% Relationship between MVs and target

# plot_relations(df, feats_cat, file_name='MV_targ.png')
#%%
feats_to_del = [191, 201, 213, 215, 224]   # features to delete
feats_to_del = [f'Var{col}' for col in feats_to_del]

feats_cat = list(filter(lambda x: x not in feats_to_del, feats_cat))
df.drop(feats_to_del, axis=1, inplace=True)
df_test.drop(feats_to_del, axis=1, inplace=True)

print('\nCategorical data. Creating new features replacing nan values with "rare" category')
print('shape of df_train:', df.shape)
print('shape of df_test:', df_test.shape)
#%% Relationship between MVs and target. Partially deleted feats.

# plot_relations(df, feats_cat, file_name='MV_targ2.png')
#%% MCNAR: using these feats create new feat: "isna_Var194" and in original feat replace feat with 'rare' - category.

feats_to_new = [194, 197, 199, 200, 206, 214, 223, 225, 229]
feats_to_new = [f'Var{col}' for col in feats_to_new]

for feat in feats_to_new:
    
    new = 'isna_' + feat
    feats_cat.append(new)
    
    df[new] = np.where(df[feat].isna(), 'missing', 'existing')
    df[feat] = np.where(df[feat].isna(), 'rare', df[feat])
    
    df_test[new] = np.where(df_test[feat].isna(), 'missing', 'existing')
    df_test[feat] = np.where(df_test[feat].isna(), 'rare', df_test[feat])

print('\nCategorical data. Creating new features replacing nan values with "rare" category')
print('shape of df_train:', df.shape)
print('shape of df_test:', df_test.shape)    

#%% MCAR: inpute 'missing'
mask = df[feats_cat].isna().any()
feats_cat_na = df[feats_cat].columns[mask]    
#%%  Relationship between MVs and target. Partially deleted feats.
# plot_relations(df, feats_cat_na, ncols=4, file_name='MV_targ3.png') 
#%% MCAR: most frequent strategy

for feat in feats_cat:

    top = df[feat].value_counts().nlargest(1).index.values[0]
    df[feat].fillna(top, inplace=True)
    df_test[feat].fillna(top, inplace=True)

print('\nCategorical data. MCAR: most_frequent')
print('shape of df_train:', df.shape)
print('shape of df_test:', df_test.shape)        
#%%
isna = df[feats_num+feats_cat+ ['labels']].isna().any().any() # False
print('train is NaN:', isna)
isna = df_test[feats_num+feats_cat+ ['labels']].isna().any().any() # False
print('test is NaN:', isna)
#%%
df[trg] = df[trg].astype(int)
df_test[trg] = df_test[trg].astype(int)
#%% save my train and test set for further processing
train_set_filled = {
    'df_train': df[feats_num+feats_cat+['labels']],
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }
with open('pkl/df_train_filled.pkl', 'wb') as file:
    pkl.dump(train_set_filled, file)

test_set_filled = {
    'df_test': df[feats_num+feats_cat+['labels']],
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }
with open('pkl/df_test_filled.pkl', 'wb') as file:
    pkl.dump(test_set_filled, file)
#%%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# tree = DecisionTreeClassifier(random_state=SEED)
# rf = RandomForestClassifier(random_state=SEED)
# gb = GradientBoostingClassifier(random_state=SEED)

# models = {'tree': tree, 'rf': rf, 'gb': gb}

# #%% Undersample data

# df_churn = df.loc[df[trg] == 1, :]
# n_samples = df_churn.shape[0]
# df_nchurn = df.loc[df[trg] == -1, :].sample(n_samples)
# df_small = pd.concat([df_churn, df_nchurn], axis=0)
# df_small = df_small.reindex(np.random.permutation(df_small.index))

# X_small, y_small = df_small.loc[:, feats_cat], df_small.loc[:, trg].astype(int)

# t = df_small[trg]
# #%%
# import shap 
# from sklearn.preprocessing import OrdinalEncoder

# X_cleaned = OrdinalEncoder().fit_transform(X_small)

# for title, model in models.items():
    
#     model.fit(X_cleaned, y_small)
#     explainer = shap.TreeExplainer(model).shap_values(X_cleaned)
#     fig = shap.summary_plot(explainer, X_cleaned, feature_names=feats_cat,
#                       max_display=len(feats_cat), show=False)                                                  
    
#     plt.savefig(f'media/shap_values_cat_{title}')





    
    
    
    
    
    
    
    
    
    
    











