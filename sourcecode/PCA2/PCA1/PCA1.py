import numpy as np
import pandas as pd

data_path = '../input/'
train = pd.read_csv(data_path + 'train.csv')

from sklearn.cross_validation import train_test_split

x,y,z=train.iloc[:,3].values,train.iloc[:,4],train.iloc[:,5];
