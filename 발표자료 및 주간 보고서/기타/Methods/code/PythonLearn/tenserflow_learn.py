import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split

# Random Forest Sample Code

# 0. Loading csv data
print('Loading data...')
data_path = 'D:/musicData/'
#train = pd.read_csv(data_path + 'train.csv')
train = pd.read_csv(data_path + 'train.csv',nrows=5000)
#test = pd.read_csv(data_path + 'train.csv',nrows=50000)

# Convert the three columns to analyze into strings
#train=train.drop(['msno','song_id'])
train['source_system_tab']=train['source_system_tab'].astype(str)
train['source_screen_name']=train['source_screen_name'].astype(str)
train['source_type']=train['source_type'].astype(str)
train['target']=train['target'].astype(int)
train, test = train_test_split(train, test_size=0.2)

# Define the column names for the data sets.
COLUMNS = ['source_system_tab','source_screen_name','source_type']
LABEL_COLUMN = 'target'
CATEGORICAL_COLUMNS = ['source_system_tab','source_screen_name','source_type']
CONTINUOUS_COLUMNS = []

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(categorical_cols.items())#dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(train)
def test_input_fn():
  return input_fn(test)


print('Configuring Model...')
# 1. Configuring Forest height parameter
hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=3, max_nodes=1000, num_classes=3, num_features=4)

# 2. Forest Estimator
classifier = tf.contrib.tensor_forest.random_forest.TensorForestEstimator(hparams)


print('Fitting data...')
classifier.fit(input_fn=train_input_fn, steps=100)
print('Evaluating data...')
print(classifier.evaluate(input_fn=test_input_fn, steps=10))