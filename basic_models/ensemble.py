import mlp_train
#import random_forest
import linear_regression
import decisionTree

import pandas
from sklearn import linear_model
import math
from statistics import mean

df = pandas.read_csv("preprocessed_data.csv")

split_index = math.floor((df.shape[0] * (4/5)))

df_dev = df.iloc[:split_index,:]
df_test = df.iloc[split_index:,:]

test_data = [[0.08614639719282975,0.7567697100407381,0.005129799471475206,0.0,0.006993006993006993,0.3453030498549394]]
linear_regression.regr.predict(test_data)
decisionTree.dt.predict(test_data)