import numpy as np
import pandas as pd
import lightgbm as lgb

print('Loading data...')
data_path = 'C:/Users/wnlwn/Desktop/input/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv',)
members = pd.read_csv(data_path + 'members.csv')
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

print('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on='song_id', how='left')
test = test.merge(songs_extra, on='song_id', how='left')

import gc

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

X=train[train.song_year != 2017]
y = X['target'].values
X = X.drop(['target'], axis=1)



#X = X.drop(['song_year'],axis=1)
#X = X.drop(['genre_ids'],axis=1)
#X = X.drop(['gender'],axis=1)
#X = X.drop(['bd'],axis=1)
#X = X.drop(['city'],axis=1)

X_test = test.drop(['id'], axis=1)


#X_test = test.drop(['genre_ids'], axis=1)
#X_test = test.drop(['gender'], axis=1)
#X_test = test.drop(['bd'], axis=1)
#X_test = test.drop(['city'], axis=1)
ids = test['id'].values

d_train = lgb.Dataset(X, y)

A = train[train.song_year == 2017]
b = A['target'].values
A = A.drop(['target'],axis=1)
#A = A.drop(['song_year'],axis=1)


#A = A.drop(['genre_ids'],axis=1)
#A = A.drop(['gender'],axis=1)
#A = A.drop(['city'],axis=1)
#A = A.drop(['bd'],axis=1)
d_validation = lgb.Dataset(A,b)

watchlist = [d_validation]

print('Training LGBM model...')

params = {}
params['learning_rate'] = 0.2
params['application'] = 'binary'
params['max_depth'] = 25
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'
model = lgb.train(params, train_set=d_train, num_boost_round=300, valid_sets=watchlist, \
verbose_eval=5)
print('Making predictions and saving them...')
p_test = model.predict(X_test)
subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')