import pandas
from sklearn import linear_model
import math
from statistics import mean

df = pandas.read_csv("preprocessed_data.csv")

split_index = math.floor((df.shape[0] * (4/5)))

df_dev = df.iloc[:split_index,:]
df_test = df.iloc[split_index:,:]

X = df_dev[['temperature','humidity','wind_speed','day','10minuteofday','prev_aggregated_consumption']]
y = df_dev['aggregated_consumption']

regr = linear_model.LinearRegression()
regr.fit(X.values, y.values)

diff = []

for index, row in df_test.iterrows():
    predict = regr.predict([[row['temperature'],row['humidity'],row['wind_speed'],row['day'],row['10minuteofday'],row['prev_aggregated_consumption']]])
    actual = row['aggregated_consumption']
    #print(f'predict: {predict[0]}, acutal: {actual}, difference: {actual - predict[0]}')
    accuracy = abs(actual - predict[0]) / actual
    diff.append(accuracy)

print(mean(diff))