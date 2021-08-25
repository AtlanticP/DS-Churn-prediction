import os
import numpy as np
import pandas as pd
import missingno as msno
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import preprocessing as prp
SEED = 32
#%%
PATH_DIR = '../../../bases/telecom-clients-prediction2'
FILE_NAME = 'orange_small_churn_train_data.csv'
PATH_FILE = os.path.join(PATH_DIR, FILE_NAME)
#%%
df = pd.read_csv(PATH_FILE, index_col='ID')
feats_num = df.columns[:190]
feats_cat = df.columns[190:-1]
trg = df.columns[-1]
print(df.shape)
#%% drop elements where target is nan
df.drop(np.where(df.iloc[:, -1].isna())[0], axis=0, inplace=True)

df.iloc[:, -1].value_counts().to_dict() 
#{-1.0: 16921, 1.0: 1377}

(df.iloc[:, -1].value_counts()/len(df)).round(3).to_dict()
# {-1.0: 0.925, 1.0: 0.075}
#%% Missing values
fig = msno.matrix(df)
fig = fig.get_figure()
fig.savefig('/home/atl/Pictures/msno.png') 
#%% HANDLING NUMERICAL MIS. DATA. Remove completly nan columns 
feats_num = df[feats_num].dropna(how='all', axis=1).columns
feats_cat = df[feats_cat].dropna(how='all', axis=1).columns
#%% Get feats with nan less than 0.925

mask = (df[feats_num].isna().mean() <= 0.925)
feats_num_naless0925 = mask[mask].index
#%% Random Sample Imputation

feats_g = [] # feats with great number of MV

for feat in feats_num_naless0925:
    
    n = df[feat].isna().sum()
    
    try:        
    
        sample = df[feat].dropna().sample(n=n, random_state=SEED)
        sample.index = df[df[feat].isna()].index
        df.loc[df[feat].isna(), feat] = sample
    
    except ValueError:

        feats_g.append(feat)   

# remove feats that raised exception    
feats_num = list(set(feats_num_naless0925)-set(feats_g))
#%% HANDLING CATEGORICAL MIS. DATA. 

feats_cat_na = df[feats_cat].columns[df[feats_cat].isna().any()]
df[feats_cat_na].isna().mean().plot.bar()

#%%
# n = len(feats_cat_na)
# fig, axes = plt.subplots(nrows=int(np.ceil(n/4)), ncols=4,
#                        figsize=(15, n))

# for idx, feat in enumerate(feats_cat_na):
    
#     df['temp'] = np.where(df[feat].isna(), 1, 0)  
#     df.groupby(trg)['temp'].mean().plot(kind='bar', ax=axes[idx//4, idx%4], title=feat)

# df.drop('temp', axis=1, inplace=True) 
# fig.tight_layout()
# fig.savefig('/home/atl/Pictures/cat_nans_targ.png')   
# fig.set_title('Distribu')
#%% remove features where nan more than 10 percent
thresh = len(df)*0.9
feats_cat_na = df[feats_cat_na].dropna(thresh=thresh, axis=1).columns
df[feats_cat_na].isna().mean().plot.bar()
#%% Create new category for missing values
feat = feats_cat_na[0]
feats_cat = []
for feat in feats_cat_na:
    df[feat] = np.where(df[feat].isna(), 'Rare', df[feat])
    feats_cat.append(feat)
#%% 
df[feats_num+feats_cat].isna().any().any() # False










    
    
    
    
    
    
    
    
    
    
    











