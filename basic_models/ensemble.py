import mlp_train
#import random_forest
import linear_regression
import decisionTree
import statistics

import pandas
from sklearn import linear_model
import math
from statistics import mean
import matplotlib.pyplot as plt

df = pandas.read_csv("preprocessed_data_test.csv")

mlp_model = mlp_train.main()

accuracies = []
labels = []
predictions = []

for index, row in df.iterrows():
    # print('|||||||||||||||||||||')
    # print(f'- Prediction number {index} -')
    test_row = row.values[:-1]
    dectre_pred = decisionTree.dt.predict([test_row.tolist()])
    test_row_2 = test_row[1:]
    linreg_pred = linear_regression.regr.predict([test_row_2.tolist()])
    test_row_3_temp = df.iloc[[index]]
    # print(test_row_3_temp.values)

    # mlp_df = df.iloc[:1,:]
    # print(mlp_df.values)

    test_row_3 = mlp_train.FeatureData(test_row_3_temp.values)
    mlp_results = mlp_train.evaluate(mlp_model, test_row_3, 1, 'cpu')
    neunet_pred = mlp_results['pred'][0].item() * 10000

    #average = statistics.mean([linreg_pred[0], dectre_pred[0], neunet_pred])
    average = statistics.mean([neunet_pred])

    # print(f'Mean of all three predictions: {average}')
    # print(f'Actual value: {row.values[-1]}')
    # print(f'Accuracy: {100 * abs(row.values[-1] - average) / row.values[-1]} %')
    # print('   ')

    accuracies.append(abs(row.values[-1] - average) / row.values[-1])
    labels.append(row.values[-1])
    predictions.append(average)

print(f' This is the average accuracies of our ensemble model: {mean(accuracies)}')

plt.figure(figsize=(10, 10))

# Plotting the graph
plt.scatter(labels, predictions, marker='o', color='blue', s=2)
plt.plot([0, max(labels)], [0, max(predictions)], color='red', linestyle='--')

# Adding labels and title
plt.xlabel('Labels')
plt.ylabel('Predictions')
plt.title('Neural Network')



# Display the plot
plt.show()


# test_data = [[0.08614639719282975,0.7567697100407381,0.005129799471475206,0.0,0.006993006993006993,0.3453030498549394]]
# test_data_2 = [[2,0.08614639719282975,0.7567697100407381,0.005129799471475206,0.0,0.006993006993006993,0.3453030498549394]]
# linreg_pred = linear_regression.regr.predict(test_data)
# dectre_pred = decisionTree.dt.predict(test_data_2)



# mlp_model = mlp_train.main()
# # mlp_train_data, mlp_dev_data, mlp_test_data = mlp_train.load_data("preprocessed_data_test.csv", 1)
# # print(mlp_test_data)
# mlp_df = df.iloc[:1,:]
# print(mlp_df.values)

# mlp_input = mlp_train.FeatureData(mlp_df.values)
# mlp_results = mlp_train.evaluate(mlp_model, mlp_input, 64, 'cpu')
# #print(mlp_results)

# # for i, result in enumerate(mlp_results):
# #     print(mlp_results['pred'][i].item() * 10000)

# neunet_pred = mlp_results['pred'][0].item() * 10000


# print('========= FINAL =========')
# print(f'LinReg: {linreg_pred[0]}')
# print(f'DecTre: {dectre_pred[0]}')
# print(f'NeuNet: {neunet_pred}')

# average = statistics.mean([linreg_pred[0], dectre_pred[0], neunet_pred])
# print('')
# print(f'Mean: {average}')
