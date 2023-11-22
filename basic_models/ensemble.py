import mlp_train
#import random_forest
import linear_regression
import decisionTree
import statistics

import pandas
from sklearn import linear_model
import math
from statistics import mean

df = pandas.read_csv("preprocessed_data_test.csv")

split_index = math.floor((df.shape[0] * (4/5)))

df_dev = df.iloc[:split_index,:]
df_test = df.iloc[split_index:,:]

test_data = [[0.08614639719282975,0.7567697100407381,0.005129799471475206,0.0,0.006993006993006993,0.3453030498549394]]
test_data_2 = [[2,0.08614639719282975,0.7567697100407381,0.005129799471475206,0.0,0.006993006993006993,0.3453030498549394]]
linreg_pred = linear_regression.regr.predict(test_data)
dectre_pred = decisionTree.dt.predict(test_data_2)

mlp_df = df.iloc[:1,:]
print(mlp_df.values)
mlp_model = mlp_train.main()
mlp_train_data, mlp_dev_data, mlp_test_data = mlp_train.load_data("preprocessed_data_test.csv", 1)
print(mlp_test_data)

#mlp_input = mlp_train.FeatureData(mlp_test_data)
mlp_results = mlp_train.evaluate(mlp_model, mlp_test_data, 64, 'cpu')
#print(mlp_results)

for i, result in enumerate(mlp_results):
    print(mlp_results['pred'][i].item() * 10000)

neunet_pred = mlp_results['pred'][0].item() * 10000


print('========= FINAL =========')
print(f'LinReg: {linreg_pred[0]}')
print(f'DecTre: {dectre_pred[0]}')
print(f'NeuNet: {neunet_pred}')

average = statistics.mean([linreg_pred[0], dectre_pred[0], neunet_pred])
print('')
print(f'Mean: {average}')
