import pandas as pd 
#%%

dct = {}
models = [
    'd1 rf: be, stnd',
    'd1 lgbm: be, stnd',
    'd1 rf',
    'd1 lgbm: ohe_t10, stnd',
    'd1 sgd: ohe_t10, stnd',
    'd1 knn: ohe_t10, stnd',
    'd1 svm: ohe_t10, stnd',
    'd1 rf: le (max_val)',
    'd1 lgbm: le (max_val)',
    'd1 rf: le(max_val), stnd',
    'd1 lgbm: le (max val), stnd',
    ]

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
'''
split = lst.split('\n')
index = split[1].split()

for model, line in zip(models, split[2:]):
    
    scores = [float(i) for i in line.split()]
    dct[model] = scores

df = pd.DataFrame(dct, index=index)
df = df.T.sort_values(by='roc_auc', ascending=False).round(3)
print(df) 