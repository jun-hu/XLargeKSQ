import numpy as np
import pandas as pd
import tensorflow as tf
import math

from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

def dense_linear(input_tensor,output_neurons):
    return tf.layers.dense(inputs=input_tensor, units=output_neurons)

def dense_dropout(input_tensor,output_neurons,mode):
    dense = tf.layers.dense(inputs=input_tensor, units=output_neurons, activation=tf.nn.relu)
    #dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    return dense

def model_fn(features, labels, mode):
    # The model
    input_layer = tf.cast(features['x'], tf.float32)
    #layer_source=tf.slice(input_layer,[0,0],[-1,42])
    #layer_genre=
    #layer_city=
    #layer_lang=
    #layer_regvia=
    #layer_else=tf.slice(input_layer,[0,42],[-1,-1])
    #l1_source=dense_dropout(layer_source,4,mode)
    #l1 = tf.concat([l1_source,layer_else],axis=1)

    l1 = dense_dropout(input_layer,30,mode)
    #l1=tf.Print(l1,[tf.shape(l1)])
    #mid = dense_dropout(l1,15,mode)
    l2 = dense_dropout(l1,8,mode)
    # Final layer for binary classification
    logits = dense_linear(l2,2)
    predictions = {
            "classes": tf.argmax(input=logits, axis=1), # Prediction
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")} # Probabilities in softmax
    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2) # Change label into onehot encoding (0->[1,0], 1->[0,1])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits) # Calculate the loss

    # Train: Neuron Optimization
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation: Accuracy
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)








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

# Load training and eval data
#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#train_data = mnist.train.images    # Returns np.array
#train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#eval_data = mnist.test.images    # Returns np.array
#eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Create the Estimator
classifier = tf.estimator.Estimator(
        model_fn=model_fn)

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=1000,
        num_epochs=None,
        shuffle=False)

classifier.train(
        input_fn=train_input_fn,
        steps=100000,
        hooks=[logging_hook])

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

'''
# Phase 5: Prediction
print('Making predictions and saving them...')
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    num_epochs=1,
    shuffle=False)
result = classifier.predict(input_fn=predict_input_fn)

subm = pd.DataFrame(result)
subm['id'] = ids
#subm['target']=subm['classes']
#subm[['id','target']].to_csv('submissionNNCl.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
subm['target']=subm['probabilities'].apply(lambda x:x[1])
subm[['id','target']].to_csv(result_path+'7submissionNNPr.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')
'''