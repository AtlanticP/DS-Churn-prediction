import pandas as pd
import numpy as np
SEED = 32
#%%
from sklearn.base import TransformerMixin, BaseEstimator

class OHE_topN(TransformerMixin, BaseEstimator):
    
    def __init__(self, top_n=10):
        
        self.top_n = top_n
        self.feats_labels = {}
        
    def convert_df(self, X):
        
        # check df is DataFrame, and covert feats type to string
        if not isinstance(X, pd.DataFrame):
            self.feats = list(map(str, range(X.shape[1])))
            df = pd.DataFrame(X, columns=self.feats)
        
        else:
            df = X
            self.feats = df.columns
        
        if isinstance(df.columns, pd.RangeIndex):
            df.columns = self.feats
            
        return df
        
    def fit(self, X, y=None):
        
        df = self.convert_df(X)  
               
        for feat in self.feats:
            
            # getting labels with largest frequencies in a feat
            lst_top = df[feat].value_counts().nlargest(self.top_n).index
            self.feats_labels[feat] = lst_top
    
        return self
    
    def transform(self, X):
        
        df = self.convert_df(X)
        
        for feat, labels in self.feats_labels.items():
        
            for label in labels:
                
                feat_new = feat + f'_{label}'
                
                df[feat_new] = np.where(df[feat] == label, 1, 0)
                
        df.drop(self.feats, axis=1, inplace=True) 
    
        return df
    
    def fit_transformer(self, X, y=None):
        
        self.fit(X)
        return self.transform(X)
#%%
# lst = list('abcdef')
# n = len(lst)

# np.random.seed(SEED)
# idx = np.random.randint(n, size=(5, 3))
# X = np.array(lst)[idx]
# dft = pd.DataFrame(X)
# dft.columns = [f'col{i}' for i in range(dft.shape[1])]

# X = dft.values
# X = dft.copy()
# ohe = OHE_topN(top_n=3)    
# # ohe.fit_transform(X, y=None)
# t = ohe.fit_transform(X)
# # t = ohe.feats_labels
# print(t)    