import tensorflow as tf
import pandas as pd

def next_batch(dataframe, series, size, start):
    return dataframe.iloc[range(start,start+size)].as_matrix(), series.iloc[range(start,start+size)].as_matrix()

# 0. Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
train = pd.read_csv(data_path + 'train.csv')#,nrows=20000,skiprows=range(1, 6000001))
#test = pd.read_csv(data_path + 'test.csv')#,nrows=500000,skiprows=range(1, 7000001))
songs = pd.read_csv(data_path + 'songs_new_normalized.csv')
songs = songs[['song_id','song_length','song_year','genre_437','genre_958','genre_2122','genre_2130','genre_1616','genre_458']]

#members = pd.read_csv(data_path + 'members.csv')
print('Data preprocessing...')
# Phase 0-1: Joining song table
#train = train.merge(members, on='msno', how='left')

# Convert the three columns to analyze into strings
#train=train.drop(['msno','song_id'],axis=1)
train['source_system_tab']=train['source_system_tab'].astype(str)
train['source_screen_name']=train['source_screen_name'].astype(str)
train['source_type']=train['source_type'].astype(str)
#train['target']=train['target'].astype(int)
#train['bd']=train['bd'].astype(int)
#train['noage']=train['bd'].apply(lambda x: 0 if x>0 and x<80 else 1)
#train['bd']=train['bd'].apply(lambda x: x/80.0 if x>0 and x<80 else 0)

dsst=pd.get_dummies(train['source_system_tab'])
dssn=pd.get_dummies(train['source_screen_name'])
dst=pd.get_dummies(train['source_type'])
dsst.columns = dsst.columns.map(lambda x: str(x) + '_sst')
dssn.columns = dssn.columns.map(lambda x: str(x) + '_ssn')
dst.columns = dst.columns.map(lambda x: str(x) + '_st')

train=train.drop(['source_system_tab','source_screen_name','source_type'],axis=1)
train=train.join(dsst).join(dssn).join(dst).merge(songs, on='song_id', how='left')
train=train.drop(['msno','song_id'],axis=1)
#train=dsst.join(dssn).join(dst).join(train['target']).join(train['bd']).join(train['noage'])
#train=train[['listen with_sst','my library_sst','radio_sst','Discover Feature_ssn','Discover Genre_ssn','Local playlist more_ssn','My library_ssn','Others profile more_ssn','Radio_ssn','Search Home_ssn','Unknown_ssn','listen-with_st','local-library_st','local-playlist_st','radio_st','target']]
#'listen with_sst','my library_sst','radio_sst','Discover Feature_ssn','Discover Genre_ssn','Local playlist more_ssn','My library_ssn','Others profile more_ssn','Radio_ssn','Search Home_ssn','Unknown_ssn','listen-with_st','local-library_st','local-playlist_st','radio_st'
#train, test = train_test_split(train, test_size=0.2)
train_t = pd.DataFrame(train['target'].astype(int))
train_t['target2']=train_t['target'].apply(lambda x:1-x)
train=train.drop(['target'],axis=1)
print(train.columns)
print(train.head(20))
print(train_t)
#test_t = test['target']
#test=test.drop(['target'],axis=1)

dimension=52#15
middle = 15
#middle2 = 4
final = 2



a_0 = tf.placeholder(tf.float32, [None, dimension])
y = tf.placeholder(tf.float32, [None, final])

w_1 = tf.Variable(tf.truncated_normal([dimension, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
#w_2 = tf.Variable(tf.truncated_normal([middle, middle2]))
#b_2 = tf.Variable(tf.truncated_normal([1, middle2]))
w_3 = tf.Variable(tf.truncated_normal([middle, final]))
b_3 = tf.Variable(tf.truncated_normal([1, final]))

z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = tf.sigmoid(z_1)
#z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
#a_2 = tf.sigmoid(z_2)
z_3 = tf.add(tf.matmul(a_1, w_3), b_3)
a_3 = tf.sigmoid(z_3)

#diff = tf.abs(tf.subtract(a_3, tf.transpose(y,(1,0))))
diff = tf.subtract(a_3, y)
cost = tf.multiply(diff, diff)
#acct_res = tf.reduce_mean(diff)

step = tf.train.AdamOptimizer(0.0001).minimize(cost)

#a_3=tf.Print(a_3, [a_3], message="a_3: ")
#w_1_print=tf.Print(w_1, [w_1], message="w_1: ")
#y2 = tf.subtract(tf.constant(1.0), y)
acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(73770):
    #batch_xs, batch_ys = mnist.train.next_batch(10)
    batch_xs, batch_ys = next_batch(train,train_t,100,i*100)
    #print(batch_xs, batch_ys)
    printout=sess.run(step, feed_dict = {a_0: batch_xs,
                                y : batch_ys})
    if i % 1000 == 0:
        val_xs, val_ys = next_batch(train,train_t,1000,i*100)
        i=i+10
        res = sess.run(acct_res, feed_dict =
                       {a_0: val_xs,
                        y : val_ys})
        #print (printout)
        print (res)