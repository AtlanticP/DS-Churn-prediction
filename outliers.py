import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.base import TransformerMixin, BaseEstimator
plt.style.use('seaborn')
#%%
class OUT_handler(TransformerMixin, BaseEstimator):
    
    def __init__(self, method='iqr'):    
       
        self.method = method 
        
    def fit(self, sample, y=None):
        
        if not isinstance(sample, pd.Series):
            sample = pd.Series(sample)
        
        # IQR 
        if self.method == 'iqr':
        
            q1 = sample.quantile(0.25)
            q3 = sample.quantile(0.75)
            iqr = q3 - q1
            self.params = {
                'iqr': iqr,
                'q1': q1,
                'q3': q3
                }
            
        elif self.method == 't_score':
            
            n = sample.shape[0]
            
            self.params = {
                'n' : n,
                'mean' : sample.mean(),
                'std' : sample.std(ddof=n-1)
                }
            
        elif self.method == 'z_score':
            
            n = sample.shape[0]
            
            self.params = {
                'n' : n,
                'mean' : sample.mean(),
                'std' : sample.std(ddof=n-1)
                }             
                
        else:
            erMes = 'wrong parameter "method". Must be "iqr", "t_score" or "z_score"'
            raise AttributeError(erMes)            
        
        return self
    
       
    def transform(self, sample, alpha=0.01):
        
        if not isinstance(sample, pd.Series):
            sample = pd.Series(sample)
                            
        if self.method == 'iqr':
            low = self.params['q1'] - 1.5*self.params['iqr']
            upper = self.params['q3'] + 1.5*self.params['iqr']
                    
                # handling outliers
            sample = np.select(
                [sample<low, sample>upper],
                [low, upper], default=sample
                )

        if self.method == 't_score':
            
            std = self.params['std']
            mean = self.params['mean']
            n = self.params['n']
            
            t = sts.t(df=n-1).isf(alpha/2)
            delta = t*(std/np.sqrt(n))
            upper = mean + delta
            low = mean - delta
            
            sample = np.select(
                [sample<low, sample>upper], 
                [low, upper], 
                sample
                )            
            
        if self.method == 'z_score':
            
            std = self.params['std']
            mean = self.params['mean']
            n = self.params['n']
            
            t = sts.t(df=n-1).isf(alpha/2)
            z = sts.norm().isf(alpha/2)
            mean = sample.mean()
            std = sample.std(ddof=n-1)
            delta = z*(std/np.sqrt(n))
            upper = mean + delta
            low = mean - delta
            
            sample = np.select(
                [sample<low, sample>upper], 
                [low, upper], 
                sample
                )
            
        return sample
#%% toy data

# SEED = 32
# shape = (1000, 3)
# np.random.seed(SEED)
# X = 10+np.random.randn(shape[0], shape[1]).round(2)
# rvs = sts.skewnorm(10, 1, 0.8).rvs(shape[0]).reshape(-1, 1)
# X = np.c_[rvs, X]
# shape = X.shape
# # outliers
# mask = np.random.choice([True, False], size=shape, p=[0.002, 0.998])
# X[mask] = np.random.randint(-3, 0) 
# # features
# feats = [f'Var{i}' for i in range(1, shape[1]+1)]
# df = pd.DataFrame(X, columns=feats)

# y = np.random.randint(2, size=shape[0]).reshape(-1, 1)
# y = pd.DataFrame(y)

# sample = df.iloc[:, 0].copy()
# fig, axes = plt.subplots(1, 2, figsize=(7, 5))
# axes[0].boxplot(sample)

# handler = OUT_handler(method='z_score')
# handler.fit(sample.values)
# sample = handler.transform(sample, alpha=0.01)  
# axes[1].boxplot(sample)
# #%%
# t = handler.params
# print(t)