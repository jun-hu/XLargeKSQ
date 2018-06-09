
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# SKLEARN
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

INPUT_DATA_PATH ="../input"
df_test = pd.read_csv(INPUT_DATA_PATH + 'test.csv',dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'},nrows=10000)
