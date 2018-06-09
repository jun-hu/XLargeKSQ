import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Random Forest Sample Code
'''
#### Load the data ####
print('Loading data...')
data_path = 'D:/musicData/'
train = pd.read_csv(data_path + 'train.csv',skiprows=range(1, 5000001))
test = pd.read_csv(data_path + 'test.csv')#,nrows=500000,skiprows=range(1, 7000001))
songs = pd.read_csv(data_path + 'songs_new_normalized.csv')
members = pd.read_csv(data_path + 'members.csv')
print('Data preprocessing...')
train = train.merge(songs, on='song_id', how='left').merge(members, on='msno', how='left')

# Convert the three columns to analyze into strings
train=train.drop(['msno','song_id'],axis=1)
train['source_system_tab']=train['source_system_tab'].astype(str)
train['source_screen_name']=train['source_screen_name'].astype(str)
train['source_type']=train['source_type'].astype(str)
train['target']=train['target'].astype(int)
train['bd']=train['bd'].astype(int)
#train['noage']=train['bd'].apply(lambda x: 0 if x>0 and x<80 else 1)
train['bd']=train['bd'].apply(lambda x: x if x>0 and x<80 else 0)

dsst=pd.get_dummies(train['source_system_tab'])
dssn=pd.get_dummies(train['source_screen_name'])
dst=pd.get_dummies(train['source_type'])
dsst.columns = dsst.columns.map(lambda x: str(x) + '_sst')
dssn.columns = dssn.columns.map(lambda x: str(x) + '_ssn')
dst.columns = dst.columns.map(lambda x: str(x) + '_st')
train=dsst.join(dssn).join(dst).join(train['target']).join(train['bd'])#.join(train['noage'])
#print(train)


# Phase 0-1: Joining song table
test = test.merge(songs, on='song_id', how='left').merge(members, on='msno', how='left')
ids = test['id'].values

# Convert the three columns to analyze into strings
test=test.drop(['msno','song_id'],axis=1)
test['source_system_tab']=test['source_system_tab'].astype(str)
test['source_screen_name']=test['source_screen_name'].astype(str)
test['source_type']=test['source_type'].astype(str)
test['bd']=test['bd'].astype(int)
test['noage']=test['bd'].apply(lambda x: 0 if x>0 and x<80 else 1)
test['bd']=test['bd'].apply(lambda x: x if x>0 and x<80 else 0)

dsst=pd.get_dummies(test['source_system_tab'])
dssn=pd.get_dummies(test['source_screen_name'])
dst=pd.get_dummies(test['source_type'])
dsst.columns = dsst.columns.map(lambda x: str(x) + '_sst')
dssn.columns = dssn.columns.map(lambda x: str(x) + '_ssn')
dst.columns = dst.columns.map(lambda x: str(x) + '_st')
test=dsst.join(dssn).join(dst).join(test['bd']).join(test['noage'])
#print(train)

#### Split the data ####
train, test = train_test_split(train, test_size=0.2) # 8:2 Random Split
train_t = train['target']
train=train.drop(['target'],axis=1)
#test=test.drop(['People local_ssn', 'People global_ssn'],axis=1)#
test_t = test['target']
test=test.drop(['target'],axis=1)

'''









# 0. Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
result_path = 'D:/musicData/results/'
train = pd.read_csv(data_path + 'train.csv',skiprows=range(1, 6000001),nrows=1300000)
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs_new_normalized.csv')
songs = songs[['song_id','song_length','song_year']]#,'genre_139','genre_242','genre_359','genre_388','genre_423','genre_437','genre_444','genre_451','genre_458','genre_465','genre_691','genre_726','genre_786','genre_829','genre_873','genre_880','genre_921','genre_940','genre_947','genre_958','genre_1011','genre_1138','genre_1152','genre_1180','genre_1259','genre_1572','genre_1609','genre_1616','genre_1955','genre_2022','genre_2122','genre_2130']]
members = pd.read_csv(data_path + 'members_new_normalized.csv')
members=members[['msno','bd','gender','registration_init_time','member_duration']]
members_additional=pd.read_csv(data_path + 'members_additional.csv')
songs_additional=pd.read_csv(data_path + 'songs_additional2.csv')

print('Data preprocessing...')
train['source_system_tab']=train['source_system_tab'].astype(str)
train['source_screen_name']=train['source_screen_name'].astype(str)
train['source_type']=train['source_type'].astype(str)
dsst=pd.get_dummies(train['source_system_tab'])
dssn=pd.get_dummies(train['source_screen_name'])
dst=pd.get_dummies(train['source_type'])
dsst.columns = dsst.columns.map(lambda x: str(x) + '_sst')
dssn.columns = dssn.columns.map(lambda x: str(x) + '_ssn')
dst.columns = dst.columns.map(lambda x: str(x) + '_st')
train=train.drop(['source_system_tab','source_screen_name','source_type'],axis=1)
train=train.join(dsst).join(dssn).join(dst).merge(songs, on='song_id', how='left').merge(members, on='msno', how='left').merge(songs_additional, on='song_id', how='left').merge(members_additional, on='msno', how='left').fillna(0.5)
train=train.drop(['msno','song_id','total_listens_x','total_relistens_x','total_listens_y','total_relistens_y'],axis=1)#'relisten_rate_x','relisten_rate_y'

ids = test['id'].values
test['source_system_tab']=test['source_system_tab'].astype(str)
test['source_screen_name']=test['source_screen_name'].astype(str)
test['source_type']=test['source_type'].astype(str)
dsst=pd.get_dummies(test['source_system_tab'])
dssn=pd.get_dummies(test['source_screen_name'])
dst=pd.get_dummies(test['source_type'])
dsst.columns = dsst.columns.map(lambda x: str(x) + '_sst')
dssn.columns = dssn.columns.map(lambda x: str(x) + '_ssn')
dst.columns = dst.columns.map(lambda x: str(x) + '_st')
test=test.drop(['id','source_system_tab','source_screen_name','source_type'],axis=1)
test=test.join(dsst).join(dssn).join(dst).merge(songs, on='song_id', how='left').merge(members, on='msno', how='left').merge(songs_additional, on='song_id', how='left').merge(members_additional, on='msno', how='left').fillna(0.5)
test=test.drop(['msno','song_id','People local_ssn', 'People global_ssn','total_listens_x','total_relistens_x','total_listens_y','total_relistens_y'],axis=1)
test_data=np.array(test.as_matrix())

train, evald = train_test_split(train, test_size=0.2)
train_t = pd.Series(train['target'].astype(int))
#train_t['target2']=train_t['target'].apply(lambda x:1-x)
train=train.drop(['target'],axis=1)
train_data = np.array(train.as_matrix())
train_labels = train_t.as_matrix()

eval_t = pd.Series(evald['target'].astype(int))
#test_t['target2']=test_t['target'].apply(lambda x:1-x)
evald=evald.drop(['target'],axis=1)
eval_data = np.array(evald.as_matrix())
eval_labels = np.array(eval_t.as_matrix())

#### Using sklearn RandomForestClassifier ####
clf = RandomForestClassifier(n_estimators=15, max_depth=15, random_state=0)
print('Fitting data...')
clf.fit(train, train_t)
print(clf.feature_importances_)
print('Evaluating data...')
result=clf.predict(evald)
print(accuracy_score(result,eval_t))


# Phase 5: Prediction
print('Making predictions and saving them...')
p_test = clf.predict_proba(test)

subm = pd.DataFrame(p_test)
subm.columns=['t0','target']
subm=subm.drop('t0',axis=1)
subm.to_csv(result_path+'submissionRF.csv.gz', compression = 'gzip', index_label='id', float_format = '%.5f')
print('Done!')
