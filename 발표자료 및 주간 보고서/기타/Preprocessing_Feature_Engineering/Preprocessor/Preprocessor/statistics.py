import numpy as np
import pandas as pd


import math
'''
####### Per-song Integrated Preprocessing #######
# Phase 1: Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
songs = pd.read_csv(data_path + 'songs_new_normalized.csv')
members = pd.read_csv(data_path + 'members_new_normalized.csv')
train = pd.read_csv(data_path + 'train.csv')#,nrows=20000)
train=train.merge(songs, on='song_id', how='left').merge(members, on='msno', how='left')

print('Grouping data...')
result=train.groupby('song_id').apply(lambda x: pd.Series(dict(
    total_listens=(x.target).count(),
    total_relistens=(x.target).sum(),
    mean_age=(x.bd).sum(),
    mean_regdate=(x.registration_init_time).sum(),
    mean_duration=(x.member_duration).sum(),
    mean_relisten_age=(x.bd * x.target).sum(),
    mean_relisten_regdate=(x.registration_init_time * x.target).sum(),
    mean_relisten_duration=(x.member_duration * x.target).sum(),
    gender_anticor=(x.gender==x.target).sum(),
    gender_decided=(x.gender!=0.5).sum()
)))

result['relisten_rate']=result['total_relistens'].apply(lambda x:x+5)/result['total_listens'].apply(lambda x:x+10)
result['mean_age']=result['mean_age']/result['total_listens']
result['mean_regdate']=result['mean_regdate']/result['total_listens']
result['mean_duration']=result['mean_duration']/result['total_listens']
result['mean_relisten_age']=(result['mean_relisten_age']/result['total_relistens']).replace([np.inf, -np.inf], np.nan).fillna(result['mean_age'])
result['mean_relisten_regdate']=(result['mean_relisten_regdate']/result['total_relistens']).replace([np.inf, -np.inf], np.nan).fillna(result['mean_regdate'])
result['mean_relisten_duration']=(result['mean_relisten_duration']/result['total_relistens']).replace([np.inf, -np.inf], np.nan).fillna(result['mean_duration'])
result['gender_anticor']=result['gender_anticor']/result['gender_decided'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
result=result.drop('gender_decided',axis=1)
print(result)
result.to_csv(data_path + 'songs_additional2.csv')
result.head(50).to_csv(data_path + 'songs_additional2_firstfew.csv')



'''
def l(x):
    try:
        return (int(x)>0 and int(x)<100)
    except:
        return False
####### Per-member Integrated Preprocessing #######
# Phase 1: Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
#songs = pd.read_csv(data_path + 'songs_new_normalized.csv')
members = pd.read_csv(data_path + 'members.csv')
print(members['bd'].count())
s=members['bd'].fillna(0)
print(s[s>0].count())
'''
train = pd.read_csv(data_path + 'train.csv')#,nrows=200000)

train=train.merge(songs, on='song_id', how='left').merge(members, on='msno', how='left')

print('Grouping data...')
result=train.groupby('msno').apply(lambda x: pd.Series(dict(
    total_listens=(x.target).count(),
    total_relistens=(x.target).sum(),
    mean_song_year=(x.song_year).sum(),
    mean_song_length=(x.song_length).sum(),
    mean_relisten_song_year=(x.song_year * x.target).sum(),
    mean_relisten_song_length=(x.song_length * x.target).sum()
)))

result['relisten_rate']=result['total_relistens'].apply(lambda x:x+5)/result['total_listens'].apply(lambda x:x+10)
result['mean_song_year']=result['mean_song_year']/result['total_listens']
result['mean_song_length']=result['mean_song_length']/result['total_listens']
result['mean_relisten_song_year']=(result['mean_relisten_song_year']/result['total_relistens']).replace([np.inf, -np.inf], np.nan).fillna(result['mean_song_year'])
result['mean_relisten_song_length']=(result['mean_relisten_song_length']/result['total_relistens']).replace([np.inf, -np.inf], np.nan).fillna(result['mean_song_length'])
print(result)
result.to_csv(data_path + 'songs_aggr.csv')
'''