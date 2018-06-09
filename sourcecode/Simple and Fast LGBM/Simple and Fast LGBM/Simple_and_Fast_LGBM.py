
import numpy as np
import pandas as pd
import lightgbm as lgb
#####################################################################################################################
print('Loading data...')
data_path = 'C:/input/'
train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})#,nrows=10000
test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})
songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})
members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'})
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

#####################################################################################################################
print('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language'] #5
#이런 식으로 원하는 컬럼만 빼서 갖고 올 수 있구나
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

#등록날짜 만료날짜를 연/월/일로 분리한다.
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))#6
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))#7
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))#8

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))#9
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))#10
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))#11 #14
members = members.drop(['registration_init_time'], axis=1) #삭제

#노래isrc로 부터 연도를 추출한다.
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

#데이터 머지
train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')
train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')

import gc

#타입을 카테고리로 만든다.
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

X = train.drop(['target'], axis=1)
X=train[train.song_year!=2017]
y = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

d_train = lgb.Dataset(X, y)

A = train[train.song_year == 2017]
b = A['target'].values
A = A.drop(['target'],axis=1)
d_validation = lgb.Dataset(A,b)

watchlist2 = [d_validation]

print('Training LGBM model...')
params = {}
#Type: numeric. The shrinkage rate applied to each iteration. Lower values lowers overfitting speed, while higher values increases overfitting speed. Defaults to 0.1.
params['learning_rate'] = 0.2
#Type: character. The label application to learn. Must be either 'regression', 'binary(Classfication)', or 'lambdarank'
params['application'] = 'binary'
#limit the max depth for tree model. This is used to deal with over-fitting when #data is small. Tree still grows by leaf-wise
params['max_depth'] = 8
#num_leaves, default=31, type=int, alias=num_leaf
#number of leaves in one tree
params['num_leaves'] = 2**7
#verbosity, default=1, type=int, alias=verbose<0 = Fatal, =0 = Error (Warn), >0 = Info
params['verbosity'] = 0
#metric, default={l2 for regression}, {binary_logloss for binary classification}, {ndcg for lambdarank}, type=multi-enum, options=l1, l2, ndcg, auc, binary_logloss, binary_error ...
params['metric'] = 'auc'

#num_boost_round: num_iterations, default=100, type=int, alias=num_iteration, num_tree, num_trees, num_round, num_rounds, num_boost_round, number of boosting iterations
#valid_sets: valid_sets: list of Datasets or None, optional (default=None) List of data to be evaluated during training.
#verbose_eval: verbose_eval : bool or int, optional (default=True)
       # Requires at least one validation data.
        #If True, the eval metric on the valid set is printed at each boosting stage.
        #If int, the eval metric on the valid set is printed at every ``verbose_eval`` boosting stage.
        #The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed.
model = lgb.train(params, valid_sets=watchlist2,train_set=d_train, num_boost_round=50,verbose_eval=8)

print('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv(data_path+'submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')
print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X, y)))
print("훈련 세트 정확도: {:.3f}".format(gbrt.score(A, b)))