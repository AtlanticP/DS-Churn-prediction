import pandas as pd 
#%%

dct = {}
models = [
    'd1 rf: SS, be; rus', 
    'd1 lgbm: SS, be; rus', 
    'd1 rf; rus', 
    'd1 lgbm: SS, ohe_top10; rus', 
    'd1 sgd: SS, ohe_top10; rus', 
    'd1 knn: SS, ohe_top10; rus', 
    'd1 svm: SS, ohe_top10; rus', 
    'd1 rf: le (max_val); rus', 
    'd1 lgbm: le (max_val); rus', 
    'd1 rf: SS, le(max_val); rus', 
    'd1 lgbm: SS, le (max val); rus', 
    'd1 rf: SS, ohe_top10; rus', 
    'd2 lgbm: SS, ohe_top10; rus', 
    'd2 sgd: SS, ohe_top10; rus', 
    'd2 knn: SS, ohe_top10; rus', 
    'd2 svm: SS, ohe_top10; rus', 
    'd2 rf: be; rus', 
    'd2 lgbm: be; rus',
    'd2 rf: le(max_val); rus',    
    'd2 lgbm: le(max_val); rus',
    'd2 rf: be; rus',
    'd2 lgbm: be; rus',
    'd2 rf: le(max_val); rus',    
    'd2 lgbm: le(max_val); rus',
    'd2 rf: SS, be; rus',
    'd2 lgbm: SS, be; rus',    
    'd2 rf: SS, le(max_val); rus',    
    'd2 lgbm: SS, le(max_val); rus',  
]
#%%

lst = '''
roc_auc  recall     prec      acc
0.58908  0.5624  0.10638  0.61176
0.58146  0.5634  0.10276  0.59680
0.589   0.567  0.106  0.607
0.587   0.573  0.105  0.598
0.564   0.572  0.097  0.558
0.525   0.531  0.083  0.520
0.584   0.600  0.102  0.570
0.59850  0.58084  0.10978  0.61358
0.59462  0.62442  0.10464  0.56934
0.597   0.576  0.109  0.614
0.586   0.593  0.103  0.580
0.60648  0.58380  0.11376  0.62574
0.62616  0.62444  0.12016  0.62766
0.57578  0.65038  0.09982  0.51244
0.54188  0.57996  0.08690  0.50952
0.60778  0.65638  0.10832  0.56650
0.60134  0.58862  0.11042  0.61212
0.62074  0.65444  0.11432  0.59216
0.61326  0.63212  0.11282  0.59726
0.61344  0.66012  0.11206  0.57378
0.60134  0.58862  0.11042  0.61212
0.62074  0.65444  0.11432  0.59216
0.613   0.632  0.113  0.598
0.613   0.660  0.112  0.574
0.60780  0.60214  0.11254  0.61264
0.61534  0.64284  0.11280  0.59200
0.613   0.632  0.113  0.598
0.618   0.659  0.114  0.583
'''
#%% 
print(lst.split('\t'))
#%%
split = lst.split('\n')
index = split[1].split()

for model, line in zip(models, split[2:]):
    
    scores = [float(i) for i in line.split()]
    dct[model] = scores

df = pd.DataFrame(dct, index=index)
df = df.T.sort_values(by='roc_auc', ascending=False).round(3)
df.sort_index(inplace=True)
print(df) 