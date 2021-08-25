import os
import numpy as np
import pandas as pd
import missingno as msno
import pickle as pkl
from sklearn.model_selection import train_test_split
SEED = 32
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
print(df.shape)
#%% drop elements where target is nan
df.drop(np.where(df.iloc[:, -1].isna())[0], axis=0, inplace=True)
#%%
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values
#%% split data and save test set
X_train, X_test, y_train, y_test = train_test_split(X, y,                    
                    test_size=0.25, shuffle=True, 
                    stratify=y, random_state=SEED)

df = pd.DataFrame(np.c_[X_train, y_train], columns=feats)
df_test = pd.DataFrame(np.c_[X_test, y_test], columns=feats)

with open('df_test.pkl', 'wb') as file:
    pkl.dump(df_test, file)

#%%
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

#%% remove features where nan more than 10 percent
thresh = len(df)*0.9
feats_cat_na = df[feats_cat_na].dropna(thresh=thresh, axis=1).columns
df[feats_cat_na].isna().mean().plot.bar()
#%% Create new category for missing values

feats_cat = []
for feat in feats_cat_na:
    df[feat] = np.where(df[feat].isna(), 'Rare', df[feat])
    feats_cat.append(feat)
#%%
df[feats_num+feats_cat+ ['labels']].isna().any().any() # False
#%% save my train_set for further processing
train_set_filled = {
    'df_train': df[feats_num+feats_cat+['labels']],
    'feats_num': feats_num,
    'feats_cat': feats_cat
    }

with open('pkl/df_train_filled.pkl', 'wb') as file:
    pkl.dump(train_set_filled, file)
    
#%%









    
    
    
    
    
    
    
    
    
    
    











