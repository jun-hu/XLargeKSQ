# https://www.kaggle.com/jeru666/interesting-kkbox-insights-with-new-features
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
import re
import math
from collections import Counter

from subprocess import check_output

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_songs = pd.read_csv('../input/songs.csv')
df_members = pd.read_csv('../input/members.csv')

#First, we will merge the train and test data with the members and songs data. We can keep
# the merged data and delete the independant ones, to save memory exhaustion on this kernel.
#--- Merging dadtaframes ---
df_train_members = pd.merge(df_train, df_members, on='msno', how='inner')
df_train_merged = pd.merge(df_train_members, df_songs, on='song_id', how='outer')

df_test_members = pd.merge(df_test, df_members, on='msno', how='inner')
df_test_merged = pd.merge(df_test_members, df_songs, on='song_id', how='outer')

#--- delete unwanted dataframe ---
del df_train_members
del df_test_members

del df_songs
del df_members

#Dropping rows with missing msno values
#Upon checking the number of rows in the original train data and the merged train data, 
#they would not be the same.
print(len(df_train))
print(len(df_train_merged))

#The same case goes for the test dataframes as well.
print(len(df_test))
print(len(df_test_merged))

#This is because rows with missing msno have also been 
#included while merging which must be dropped:
df_train_merged = df_train_merged[pd.notnull(df_train_merged['msno'])]
df_test_merged = df_test_merged[pd.notnull(df_test_merged['msno'])]

#Saving target and id columns separately
df_test_merged.columns
df_train_merged.columns

#Saving the id column from test data and target column from train data separately;
# and deleting those respective columns from the dataframes.
#--- before that save unique columns in train and test set separately ---
df_train_target = df_train_merged['target'].astype(np.int8)
df_test_id = df_test_merged['id']

#--- now dropping those columns from respective dfs ---
df_train_merged.drop('target', axis=1, inplace=True)
df_test_merged.drop('id', axis=1, inplace=True)

#Appending another column is_train to distinguish between train and test data:
df_train_merged['is_train'] = 1
df_test_merged['is_train'] = 0

