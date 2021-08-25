import pickle as pkl
import numpy as np 
import pandas as pd

SEED = 32
#%%
with open('pkl/df_train_filled.pkl', 'rb') as file:
    data = pkl.load(file)

df = data['df_train']
trg = 'labels'
feats_num = data['feats_num']
feats_cat = data['feats_cat']
#%% HANDLING CATEGORICAL DATA. Cardinality.

cards = []
for feat in feats_cat:
    card = df[feat].value_counts().size
    cards.append(card)
    
cards = pd.Series(cards, index=feats_cat)   
cards.plot.bar()
#%% OHE with many categories: top-10
feats_cat_new = []

for feat in feats_cat:
    
    labels = df[feat].value_counts().nlargest(10).index
    
    for label in labels:
        
        feat_new = feat + '_' + label
        df[feat_new] = np.where(df[feat] == label, 1, 0)
        feats_cat_new.append(feat_new)
#%% save data to train model
df.drop(feats_cat, axis=1, inplace=True)

train_set_filled = {
    'df_train': df,
    'feats_num': feats_num,
    'feats_cat': feats_cat_new
    }

with open('pkl/df_train_prp.pkl', 'wb') as file:
    
    pkl.dump(train_set_filled, file)

        
        
    





















