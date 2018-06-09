import numpy as np
import pandas as pd

# 0. Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
result_path = 'D:/musicData/results/'
data= pd.read_csv(result_path + '0submissionNNPr.csv')
data['target0']=data['target']
for i in range(1,5):
    data['target'+str(i)]=pd.read_csv(result_path + str(i)+'submissionNNPr.csv')['target']
histogram=[0,0,0,0,0,0]

def rowproc(row):
    t=[row['target0'],row['target1'],row['target2'],row['target3'],row['target4']]
    t1=[i for i in t if i>=0.5]
    t2=[i for i in t if i<=0.5]
    l=len(t1)
    histogram[l]=histogram[l]+1
    return sum(t1)/len(t1) if l>=3 else sum(t2)/len(t2)

data['target']=data.apply(rowproc, axis=1)
print(histogram)
data[['id','target']].to_csv(result_path+'submissionNNVt.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')