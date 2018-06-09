import numpy as np
import pandas as pd
import gc

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
train = pd.read_csv(data_path + 'test.csv')#,nrows=1000)

# Convert the three columns to analyze into strings
train['source_system_tab']=train['source_system_tab'].astype(str)
train['source_screen_name']=train['source_screen_name'].astype(str)
train['source_type']=train['source_type'].astype(str)

#test = pd.read_csv(data_path + 'test.csv')
#songs = pd.read_csv(data_path + 'songs.csv')
#members = pd.read_csv(data_path + 'members.csv')

# Labels of source
# source_system_tab : nan, 'settings', 'explore', 'my library', 'listen with', 'notification', 'radio', 'search', 'null', 'discover'
# source_screen_name: nan, 'Discover Genre', 'Concert', 'Discover Chart', 'Search Trends', 'Payment', 'Album more', 'Discover Feature',
#                     'Unknown', 'Radio', 'Others profile more', 'Artist more', 'My library_Search', 'Search Home', 'My library', 'Local playlist more',
#                     'Discover New', 'Self profile more', 'Search', 'Online playlist more', 'Explore'
# source_type       : nan, 'top-hits-for-artist', 'artist', 'local-playlist', 'topic-article-playlist', 'local-library',
#                     'song', 'my-daily-playlist', 'album', 'online-playlist', 'song-based-playlist', 'radio', 'listen-with'

# Phase 2: Labelling Data
# Data labels are changed into responsible integer from 0 to (number of labels)-1.
# This is for the further analysis using 3d arrays and panels
print('Labeling data...')
# Function fit extracts unique labels from the data.
sst=LabelEncoder().fit(train['source_system_tab'])
ssn=LabelEncoder().fit(train['source_screen_name'])
st=LabelEncoder().fit(train['source_type'])
# Function transform generates array with values changed to the corresponding integers.
train['source_system_tab']=sst.transform(train['source_system_tab'])
train['source_screen_name']=ssn.transform(train['source_screen_name'])
train['source_type']=st.transform(train['source_type'])

# Phase 3: Counting Data
print('Counting data...')
# Two 1D arrays for each axis is prepared to maintain count by axis entry. This is used for the analysis and also sorting the axis indices.
sst_count=np.zeros(shape=(len(sst.classes_)))
ssn_count=np.zeros(shape=(len(ssn.classes_)))
st_count=np.zeros(shape=(len(st.classes_)))
sst_countp=np.zeros(shape=(len(sst.classes_)))
ssn_countp=np.zeros(shape=(len(ssn.classes_)))
st_countp=np.zeros(shape=(len(st.classes_)))
# Two 3D arrays are prepared for the actual count by three axes. One is for total, and one is for the target entry only.
positive = np.zeros(shape=(len(sst.classes_),len(ssn.classes_),len(st.classes_)))
result2 = np.zeros(shape=(len(sst.classes_),len(ssn.classes_),len(st.classes_)))
# Actual counting occurs here.
for i in range(len(train['source_system_tab'])):
    result2[train['source_system_tab'][i], train['source_screen_name'][i], train['source_type'][i]]+=1
    sst_count[train['source_system_tab'][i]]+=1
    ssn_count[train['source_screen_name'][i]]+=1
    st_count[train['source_type'][i]]+=1
    '''if train['target'][i]==1 :
        positive[train['source_system_tab'][i], train['source_screen_name'][i], train['source_type'][i]]+=1
        sst_countp[train['source_system_tab'][i]]+=1
        ssn_countp[train['source_screen_name'][i]]+=1
        st_countp[train['source_type'][i]]+=1'''
    if i%10000==0 :
        print(i)

# Phase 4: Renaming Axes & Axis Report
# The total counts of each axes entries are appended in front of the entries.
# This is the workaround as sort_index with level parameter is unimplemented.
print('Renaming Labels...')
print('')
print('Axis Reports')
print('x:source_system_tab')
for i in range(len(sst.classes_)):
    print(sst.classes_[i]+' : '+str(sst_countp[i]/sst_count[i])+' ('+str(sst_countp[i])+'/'+str(sst_count[i])+')')
    sst.classes_[i]='{:07.0f}'.format(sst_count[i])+sst.classes_[i]
print('y:source_screen_name')
for i in range(len(ssn.classes_)):
    print(ssn.classes_[i]+' : '+str(ssn_countp[i]/ssn_count[i])+' ('+str(ssn_countp[i])+'/'+str(ssn_count[i])+')')
    ssn.classes_[i]='{:07.0f}'.format(ssn_count[i])+ssn.classes_[i]
print('z:source_type')
for i in range(len(st.classes_)):
    print(st.classes_[i]+' : '+str(st_countp[i]/st_count[i])+' ('+str(st_countp[i])+'/'+str(st_count[i])+')')
    st.classes_[i]='{:07.0f}'.format(st_count[i])+st.classes_[i]
print('')

# Phase 5: Paneling
# Load the data into Panel
print('Paneling data...')
p_total = pd.Panel(result2,sst.classes_,ssn.classes_,st.classes_)
'''
# These are codes with unimplemented exceptions on level parameter
p_total.sort_index(axis=0,level=sst_count.tolist(),inplace=True)
p_total.sort_index(axis=1,level=ssn_count.tolist(),inplace=True)
p_total.sort_index(axis=2,level=st_count.tolist(),inplace=True)
'''
p_total=p_total.sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False).sort_index(axis=2,ascending=False)
p_total.to_excel('totalt.xlsx')
p_positive = pd.Panel(positive,sst.classes_,ssn.classes_,st.classes_)
p_positive.sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False).sort_index(axis=2,ascending=False).to_excel('positivet.xlsx')
# Generating ratio
# (Better code is possible, as I'm new to Python)
for i in range(len(result2)):
    for j in range(len(result2[i])):
        for k in range(len(result2[i][j])):
            if result2[i][j][k]>0:
                positive[i][j][k]=float(positive[i][j][k])/float(result2[i][j][k]);
p_ratio=pd.Panel(positive,sst.classes_,ssn.classes_,st.classes_)
p_ratio=p_ratio.sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False).sort_index(axis=2,ascending=False)
p_ratio.to_excel('ratiot.xlsx')

# Phase 6. 3d plotting test
import math
# Only plot nonzero entries
coords_nonzero=result2.nonzero()
# Dot sizes are in log scale
total_nonzero=list(map(lambda x: 100.0*math.log10(x), result2[coords_nonzero]))
# Bring the ratio generated
ratio_nonzero=positive[coords_nonzero]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Preparing colors, with 1 red and 0 purple
norm = matplotlib.colors.Normalize()
color=plt.cm.rainbow(norm(ratio_nonzero))

ax.scatter(coords_nonzero[0],coords_nonzero[1],coords_nonzero[2], s=total_nonzero, c=plt.cm.rainbow(norm(ratio_nonzero)))

# Axis Labelling
ax.set_xlabel('source_system_tab')
ax.set_ylabel('source_screen_name')
ax.set_zlabel('source_type')

# Forcing the labels to show
ax.set_xticks(np.arange(0, len(sst.classes_), 1))
ax.set_yticks(np.arange(0, len(ssn.classes_), 1))
ax.set_zticks(np.arange(0, len(st.classes_), 1))
# Setting the labels
ax.axes.set_xticklabels(list(map(lambda x: x[7:],sst.classes_)))
ax.axes.set_yticklabels(list(map(lambda x: x[7:],ssn.classes_)))
ax.axes.set_zticklabels(list(map(lambda x: x[7:],st.classes_)))

plt.show()
