import os
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
#%%

PATH_DIR = '../../../bases/telecom-clients-prediction2'
FILE_NAME = 'orange_small_churn_train_data.csv'
PATH_FILE = os.path.join(PATH_DIR, FILE_NAME)
#%%
df = pd.read_csv(PATH_FILE, index_col='ID')
feats_num = df.columns[:190]
feats_cat = df.columns[190:]
print(df.shape)
#%%
trg = df.columns[-1]
#%%
df.drop(np.where(df.iloc[:, -1].isna())[0], axis=0, inplace=True)
df.iloc[:, -1].value_counts().to_dict() #{-1.0: 16921, 1.0: 1377}

# {-1.0: 0.925, 1.0: 0.075}
(df.iloc[:, -1].value_counts()/len(df)).round(3).to_dict()
#%% Missing values
fig = msno.matrix(df)
fig = fig.get_figure()
fig.savefig('/home/atl/Pictures/msno.png') 
#%% remove columns with
feats_num = df[feats_num].dropna(how='all', axis=1).columns
feats_cat = df[feats_cat].dropna(how='all', axis=1).columns
#%%

shares_na_num = 1-df[feats_num].isna().mean()
#%% Get feats where nan more 0.925
mask = df[feats_num].isna().mean() > 0.925
feats_num_namore925 = mask[mask].index

t = df[feats_num_namore925].isna().mean()
print(t.shape)
#%% EDA with feats where nan more 0.925
n = len(feats_num_namore925)
fig, axes = plt.subplots(nrows=n//5+1, ncols=5, figsize=(15, 3*(n//5)))

for idx, feat in enumerate(feats_num_namore925[:n]):
    
    df['temp'] = np.where(df[feat].isna(), 1, 0)
    groups_mean = df.groupby(trg)['temp'].mean()
    groups_mean.plot.bar(ax=axes[idx//5, idx%5], title=feat)
    
df.drop('temp', inplace=True, axis=1)
fig.tight_layout()    
fig.savefig('/home/atl/Pictures/feats_with_nan_more0975.png')

# as we see in that features there is no any relation 
# between missing values and label variable
#%% Get feats with nan less than 0.925

mask = (df[feats_num].isna().mean() <= 0.925)
feats_num_naless0925 = mask[mask].index
n = len(feats_num_naless0925)
fig, axes = plt.subplots(nrows=n//5+1, ncols=5, figsize=(15, 3*(n//5)))

for idx, feat in enumerate(feats_num_naless0925[:n]):
    
    df['temp'] = np.where(df[feat].isna(), 1, 0)
    groups_mean = df.groupby(trg)['temp'].mean()
    groups_mean.plot.bar(ax=axes[idx//5, idx%5], title=feat)
    
df.drop('temp', inplace=True, axis=1)
fig.tight_layout()    
fig.savefig('/home/atl/Pictures/feats_with_nan_less0975.png')



    
    
    
    
    
    
    
    
    
    
    
    











