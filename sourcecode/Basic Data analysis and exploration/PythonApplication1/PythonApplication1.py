
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #draw graph, plotting
#%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#import subprocess as sp
#print(sp.check_output(["ls","../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

df_train = pd.read_csv("../input/train.csv")
df_songs = pd.read_csv("../input/songs.csv")
df_songs_extra = pd.read_csv("../input/song_extra_info.csv")
df_members = pd.read_csv("../input/members.csv",parse_dates=["registration_init_time","expiration_date"])

#간단 정보 출력
df_train.info()
df_train.head()
df_songs.head()
df_songs_extra.head()

#train데이터에 음악데이터 merge
df_train =df_train.merge(df_songs,how="left",on="song_id")
df_train =df_train.merge(df_songs_extra,how="left",on="song_id")
df_train.head()

#target 수 비교
sns.countplot(df_train['target']) 


#source_system_tab에 따른 타겟 비교, hue는 y데이터를 의미.
plt.figure(figsize=(12,10))
sns.countplot(df_train['source_system_tab'],hue=df_train['target']) 


#source_type에 따른 타겟 비교, hue는 y데이터를 의미.
plt.figure(figsize=(12,10))
g = sns.countplot(df_train['source_type'],hue=df_train['target'])
locs, labels = plt.xticks()
g.set_xticklabels(labels,rotation=45)

#dropping Null values for source_system_tab inplace를 트루로 두면 nan을 none으로 바꾼다.
df_train.dropna(subset=["source_system_tab"],inplace=True) 
#Not considering the rows where source_system_tab is settings or notifications since those counts negligible and too small
df_train = df_train.query("source_system_tab != 'settings' or source_system_tab !='notification'")
#Removing the rows where source_type is artist, topic-article-playlist or my-daily-playlist since those counts can be ignored too
df_train = df_train.query("source_type != 'artist' or source_type !='topic-article-playlist' or source_type!='my-daily-playlist'")
df_train.info()

#source_system_tab과 source_type을 카테고리 타입으로 지정
df_train['source_system_tab'] = df_train['source_system_tab'].astype("category")
df_train['source_type'] = df_train['source_type'].astype("category")
df_train.info()

#######################################################################################
#음악의 language수 체크
df_train['language'].isnull().sum()
df_train['language'].value_counts()
df_train.head()

plt.figure(figsize=(12,10))
sns.countplot(df_train['language'],hue=df_train['target'])
#Language with code 3.0 seems taiwaneese (after a bit of googling ) 52.0 is of course english 31.0 is Korean .

#language수를 비율로 체크한다.
x = df_train['language'].value_counts()

df_len = len(df_train)
for lang_id,count in zip(df_train['language'].value_counts().index,df_train['language'].value_counts()) : 
    
    print(lang_id,":",(100*count / df_len))

df_train.dropna(subset=["language"],inplace=True)
df_train.head()
df_members.info()
df_members.head()
#유저데이터 머지
df_train = df_train.merge(df_members,how="left",on="msno")

#########################################################################################
#유저나이 분석
plt.figure(figsize=(14,12))
df_train['bd'].value_counts(sort=True).plot.bar()
plt.xlim([-10,60])

#0세 이하 수 체크
len(df_train.query("bd< 0"))
#0세 이상만 데이터로 사용
df_train = df_train.query("bd >= 0")
#100세이상 수 체크
len(df_train.query("bd > 100"))
#임시로 5세이상과 80세미만의 데이터로 그래프를 그려본다.
df_train_temp = df_train.query("bd >=5 and bd <80")
plt.figure(figsize=(15,12))
sns.countplot(df_train_temp['bd'],hue=df_train_temp['target'])


#5개의 나이 그룹으로 나눠본다
df_train_temp['age_range'] = pd.cut(df_train_temp['bd'],bins=[5,10,18,30,45,60,80])
#df_train_temp.head()
plt.figure(figsize=(15,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp['target'])

##################################################################################
#장르분석
df_train_temp['genre_ids'].value_counts().head()
#성별 나눔
gender_grp = df_train_temp.groupby("gender")

#나이대와 노래길이를 고려한 타켓수
plt.figure(figsize=(15,12))
sns.boxplot(df_train_temp['age_range'],df_train_temp["song_length"]/60000,hue=df_train_temp['target'],)
plt.ylabel("Song Length in Minutes")
plt.xlabel("Age Groups")
plt.ylim([0,6])

df_train_temp.info()
df_train_temp[df_train_temp.columns[:10]].head()
df_train_temp[df_train_temp.columns[10:]].head()

#나이대별 접근 방법1(source_type)
plt.figure(figsize=(14,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_type"])
plt.legend(loc="upper right")

#나이대별 접근 방법2(source_screen_name)
plt.figure(figsize=(14,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_screen_name"])
plt.legend(loc="upper right")

#나이대별 접근 방법3(source_system_tab)
plt.figure(figsize=(14,12))
sns.countplot(df_train_temp['age_range'],hue=df_train_temp["source_system_tab"])
plt.legend(loc="upper right")

#장르별 카운트
plt.figure(figsize=(14,12))
sns.countplot(df_train['genre_ids'],hue=df_train["target"])


plt.show()