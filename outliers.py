import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.base import TransformerMixin, BaseEstimator

#%%
class OUT_handler(TransformerMixin, BaseEstimator):
    
    def __init__(self, method='iqr', log=True, res=0, plot=True):    
        
        self.method = method 
        self.log = log 
        self.res = res
        self.plot = plot 

        
    def fit(self, df, y=None):
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be pandas DataFrame object')
    
        self.params = {}    # !!!!! parameters of transformation        
        

        for idx, feat in enumerate(feats):
            
            # log transformation    
            if self.log:
        
                df.loc[:, feat] = df.loc[:, feat].apply(
                    lambda x: np.log(x+self.res)
                    )
        
            # IQR 
            if self.method == 'iqr':
            
                q1 = df[feat].quantile(0.25)
                q3 =df[feat].quantile(0.75)
                iqr = q3 - q1
                self.params[feat] = {
                    'iqr': iqr,
                    'q1': q1,
                    'q3': q3
                    }
            
            if self.method == 't_score':
                
                n = X.shape[0]
                
                self.params[feat] = {
                    'n' : n,
                    'mean' : df[feat].mean(),
                    'std' : df[feat].std(ddof=n-1)
                    }
                
        return self
    
    def transform(self, df, alpha=0.01):
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be pandas DataFrame object')
        
        feats = df.columns
        nrows = df.shape[1]
        ncols = 4
        
        if self.plot:            
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))

        for idx, feat in enumerate(feats):
            
            if self.plot:
            
                sns.boxplot(y=df.loc[:, feat] , ax=axes[idx, 0], x=y)
                axes[idx, 0].set_title(f'{feat}. With outliers')
                sns.kdeplot(x=df.loc[:, feat] , hue=y, ax=axes[idx, 1])
                axes[idx, 1].set_title(f'{feat}. With outliers')
            
            # transformation
            if self.method == 'iqr':
                low = self.params[feat]['q1'] - 1.5*self.params[feat]['iqr']
                upper = self.params[feat]['q3'] + 1.5*self.params[feat]['iqr']
                    
                # handling outliers
                df[feat] = np.select(
                    [df[feat]<low, df[feat]>upper],
                    [low, upper], default=df[feat]
                    )
            if self.method == 't_score':
                
                std = self.params[feat]['std']
                mean = self.params[feat]['mean']
                n = self.params[feat]['n']
                
                t = sts.t(df=n-1).isf(alpha/2)
                delta = t*(std/np.sqrt(n))
                upper = mean + delta
                low = mean - delta
                df[feat] = np.select(
                    [df[feat]<low, df[feat]>upper], 
                    [low, upper], 
                    df[feat]
                    )

            if self.plot:
            
                sns.boxplot(y=df[feat], ax=axes[idx, 2], x=y)
                axes[idx, 2].set_title(f'{feat}. Without outliers')
                sns.kdeplot(x=df[feat], hue=y, ax=axes[idx, 3])
                axes[idx, 3].set_title(f'{feat}. Without outliers')
                
        if self.plot:
            fig.tight_layout()
            fig.savefig('media/temp2.png')    
            
        return df
#%% toy data

shape = (1000, 3)
X = 10+np.random.randn(shape[0], shape[1]).round(2)
rvs = sts.skewnorm(10, 1, 0.8).rvs(shape[0]).reshape(-1, 1)
X = np.c_[rvs, X]
shape = X.shape
# outliers
mask = np.random.choice([True, False], size=shape, p=[0.002, 0.998])
X[mask] = np.random.randint(-3, 0) 
# features
feats = [f'Var{i}' for i in range(1, shape[1]+1)]
df = pd.DataFrame(X, columns=feats)
y = np.random.randint(2, size=shape[0])

handler = OUT_handler(method='t_score', log=False, 
                      res=0, plot=True)
handler.fit(df, y)
df = handler.transform(df)  