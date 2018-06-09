import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

import math
import datetime
def trimmer(x):
    return 0 if x<0 else 1 if x>1 else x

def normalizer(min_value, max_value, func, positive=False, unknown=0.5):
    return lambda x: unknown if math.isnan(float(x)) or (positive and x<=0) else trimmer((func(float(x))-func(max_value))/(func(min_value)-func(max_value)))

def decayer(current, halflife, unknown=0.5):
    return lambda x: unknown if math.isnan(float(x)) else 1 if float(x)>current else math.pow(0.5,(current-float(x))/halflife)

def date2int(yyyymmdd):
    return datetime.datetime.strptime(str(yyyymmdd), "%Y%m%d").timestamp()

def dummy_span(data,column_strs,prefix=""):
    print('Dummying Started for',prefix)
    result=pd.DataFrame()
    for i in column_strs:
        result[prefix+i]=data.apply(lambda x: 0 if str(x)==i else 1)
        print('- Spanning:',prefix+i)
    result[prefix+"others"]=result.sum(axis=1).apply(lambda x:1-x)
    print('Dummying Done for',prefix)
    return result

print('Loading data...')
data_path = 'D:/musicData/'
members = pd.read_csv(data_path + 'members.csv')
members['bd']=members['bd'].apply(normalizer(9,49,math.sqrt,True))
members['registration_init_time']=members['registration_init_time'].apply(date2int)
members['expiration_date']=members['expiration_date'].apply(date2int)
members['member_duration']=members['expiration_date'].subtract(members['registration_init_time'])
members=members.drop('expiration_date',axis=1)
members['gender']=members['gender'].apply(lambda x: 1 if x=='male' else 0 if x=='female' else 0.5)
members['member_duration']=members['member_duration'].apply(normalizer(1,444873600,math.log2,True))
members['registration_init_time']=members['registration_init_time'].apply(decayer(1488258000,100000000))
members.head(n=50).to_csv(data_path + 'members_new_normalized_firstfew.csv',index=False)
members.to_csv(data_path + 'members_new_normalized.csv',index=False)
print(members['registration_init_time'].max())
'''
####### Song Genre Normalization 2 #######
# Phase 1: Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
songs = pd.read_csv(data_path + 'songs_genretop32_normalized.csv')
songs['song_length']=songs['song_length'].apply(normalizer(32768,1048576,math.log2))
songs['song_year']=songs['song_year'].apply(decayer(2017,10))
songs.head(n=50).to_csv(data_path + 'songs_new_normalized_firstfew.csv',index=False)
songs.to_csv(data_path + 'songs_new_normalized.csv',index=False)
'''


'''
genre_col=['genre_139','genre_242','genre_359','genre_388','genre_423','genre_437','genre_444','genre_451','genre_458','genre_465','genre_691','genre_726','genre_786','genre_829','genre_873','genre_880','genre_921','genre_940','genre_947','genre_958','genre_1011','genre_1138','genre_1152','genre_1180','genre_1259','genre_1572','genre_1609','genre_1616','genre_1955','genre_2022','genre_2122','genre_2130']
tmp=songs[genre_col]
songs[genre_col]=tmp.div(tmp.sum(axis=1), axis=0).fillna(0) #How about 1/32?
songs=songs[['song_id','song_length','language','song_year','genre_139','genre_242','genre_359','genre_388','genre_423','genre_437','genre_444','genre_451','genre_458','genre_465','genre_691','genre_726','genre_786','genre_829','genre_873','genre_880','genre_921','genre_940','genre_947','genre_958','genre_1011','genre_1138','genre_1152','genre_1180','genre_1259','genre_1572','genre_1609','genre_1616','genre_1955','genre_2022','genre_2122','genre_2130']]
songs.to_csv(data_path + 'songs_genretop32_normalized.csv',index=False)
'''


'''
####### Song Preprocessing #######
# Phase 1: Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
songs = pd.read_csv(data_path + 'songs.csv')
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

# Phase 2: Preprocessing
print('Data preprocessing...')

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


songs = songs.merge(songs_extra, on='song_id', how='left')
songs['song_year'] = songs['isrc'].apply(isrc_to_year)

genre_id_sample=['139','242','359','388','423','437','444','451','458','465','691','726','786','829','873','880','921','940','947','958','1011','1138','1152','1180','1259','1572','1609','1616','1955','2022','2122','2130']
for i in genre_id_sample:
    songs['genre_'+i]=songs['genre_ids'].apply(lambda x: 0 if ('|'+str(x)+'|').find('|'+i+'|')==-1 else 1)
    print('Done:',i)
songs.drop(['isrc', 'name','genre_ids','artist_name','composer','lyricist'], axis=1, inplace=True)
print(songs)
#songs.head(n=37).to_csv(data_path + 'songs_genretop32_firstfew.csv',index=False)
songs.to_csv(data_path + 'songs_genretop32.csv',index=False)
'''

'''
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
# Phase 2-1: Joining song table
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

# Phase 2-2: Date Processing
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)

# Phase 2-3: Joining people table
members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')

train = train.fillna(-1)
test = test.fillna(-1)

import gc
del members, songs; gc.collect();

# Phase 3: Labeling
cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

del train, test; gc.collect();

# Phase 4: Training the model
X_train, X_valid, y_train, y_valid = train_test_split(X, y, \
    test_size=0.1, random_state = 12)
    
del X, y; gc.collect();

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


print('Training LGBM model...')
params = {}
params['learning_rate'] = 0.4
params['application'] = 'binary'
params['max_depth'] = 15
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'

model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, \
early_stopping_rounds=10, verbose_eval=10)

# Phase 5: Prediction
print('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')
'''