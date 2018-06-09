import numpy as np
import pandas as pd
import gc
import mca

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

# Phase 1: Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
#train = pd.read_csv(data_path + 'train.csv')
train = pd.read_csv(data_path + 'train.csv')#,nrows=1000)
songs = pd.read_csv(data_path + 'songs_genretop32_normalized.csv')
train=train.loc[train['target'] == 1]
train = train.merge(songs, on='song_id', how='left')
genre_col=['genre_139','genre_242','genre_359','genre_388','genre_423','genre_437','genre_444','genre_451','genre_458','genre_465','genre_691','genre_726','genre_786','genre_829','genre_873','genre_880','genre_921','genre_940','genre_947','genre_958','genre_1011','genre_1138','genre_1152','genre_1180','genre_1259','genre_1572','genre_1609','genre_1616','genre_1955','genre_2022','genre_2122','genre_2130']

print(train[genre_col].sum())

'''
print(train.shape)
train['source_system_tab']=train['source_system_tab'].astype(str)
train['source_screen_name']=train['source_screen_name'].astype(str)
train['source_type']=train['source_type'].astype(str)
train=train.drop(['msno','song_id','target'],axis=1)
print(train.shape)
mca_train = mca.MCA(train)
print (mca_train)
'''

'''
members = pd.read_csv(data_path + 'members.csv')
radio=pd.read_csv(data_path+'train_radio_merged.csv',encoding = "ISO-8859-1")

t=radio.groupby(['msno']).size()
p=radio.loc[lambda a:a.target==1, :].groupby(['msno']).size()

x=pd.concat([t,p], axis=1).fillna(0)
x.columns = ['total', 'positive']
#x=x.to_frame(name='msno')
print(t)
print(p)
print(x)
print(x.columns)
x.join(members.set_index('msno')).to_csv(data_path+'radiostat.csv')
'''
