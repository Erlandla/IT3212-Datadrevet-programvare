import pandas
from sklearn import linear_model
import math
from statistics import mean
import numpy as np
from sklearn.model_selection import train_test_split

df = pandas.read_csv("preprocessed_data.csv")

split_index = math.floor((df.shape[0] * (4/5)))

df_dev = df.iloc[:split_index,:]
df_test = df.iloc[split_index:,:]

X = df_dev[['temperature','humidity','wind_speed','day','10minuteofday','prev_aggregated_consumption']]
y = df_dev['aggregated_consumption']

labels = np.array(df['aggregated_consumption'])
df = df.drop('aggregated_consumption', axis=1)
df = df.drop(df.columns[0], axis=1)
features = np.array(df)
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

regr = linear_model.LinearRegression()
#regr.fit(X.values, y.values)
regr.fit(train_features, train_labels)

diff = []

for index, row in df_test.iterrows():
    predict = regr.predict([[row['temperature'],row['humidity'],row['wind_speed'],row['day'],row['10minuteofday'],row['prev_aggregated_consumption']]])
    actual = row['aggregated_consumption']
    #print(f'predict: {predict[0]}, acutal: {actual}, difference: {actual - predict[0]}')
    accuracy = abs(actual - predict[0]) / actual
    diff.append(accuracy)

print(mean(diff))